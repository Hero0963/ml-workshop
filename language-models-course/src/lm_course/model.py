# src/lm_course/model.py
"""A decoder-only Transformer that can be configured as GPT-2 or as a modern variant.

Lesson 04 builds the parts (attention, positions, norms, MLPs), lesson 05 assembles GPT-2 and
loads OpenAI's weights into it, lesson 06 trains the modern variants. The switches:

- ``norm``: "layernorm" (GPT-2) or "rmsnorm" (Llama, nanochat)
- ``position``: "learned" absolute embeddings (GPT-2) or "rope" rotary embeddings
- ``mlp``: "gelu" (GPT-2), "swiglu" (Llama) or "relu2" (nanochat)
- ``n_kv_head`` < ``n_head``: grouped-query attention
- ``qk_norm``, ``logit_softcap``, ``bias``, ``tie_embeddings``, ``causal``
"""

import json
import math
import struct
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

GPT2_SIZES = {
    # name: (n_layer, n_head, d_model) from the GPT-2 paper, Table 2
    "gpt2": (12, 12, 768),
    "gpt2-medium": (24, 16, 1024),
    "gpt2-large": (36, 20, 1280),
    "gpt2-xl": (48, 25, 1600),
}
GPT2_VOCAB_SIZE = 50257
GPT2_CONTEXT_LENGTH = 1024
INIT_STD = 0.02


@dataclass
class GPTConfig:
    vocab_size: int = GPT2_VOCAB_SIZE
    context_length: int = GPT2_CONTEXT_LENGTH
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    n_kv_head: int | None = None  # None: same as n_head (plain multi-head attention)
    d_ff: int | None = None  # None: 4 * d_model, or 8/3 * d_model for SwiGLU
    norm: str = "layernorm"
    norm_affine: bool = True  # learnable gain (and bias for LayerNorm)
    position: str = "learned"
    rope_base: float = 10_000.0
    mlp: str = "gelu"
    bias: bool = True
    tie_embeddings: bool = True
    qk_norm: bool = False
    logit_softcap: float | None = None
    causal: bool = (
        True  # False turns the decoder into a bidirectional encoder (lesson 13)
    )
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.n_head % self.kv_heads != 0:
            raise ValueError("n_head must be a multiple of n_kv_head")

    @property
    def kv_heads(self) -> int:
        return self.n_kv_head or self.n_head

    @property
    def ff_dim(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        if self.mlp == "swiglu":
            # keep the parameter count of a 4x MLP: 3 matrices of 8/3 d instead of 2 of 4 d
            return 64 * math.ceil(8 * self.d_model / 3 / 64)
        return 4 * self.d_model

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


def gpt2_config(size: str = "gpt2", **overrides) -> GPTConfig:
    """The architecture of OpenAI's GPT-2 (2019)."""
    n_layer, n_head, d_model = GPT2_SIZES[size]
    return replace(
        GPTConfig(n_layer=n_layer, n_head=n_head, d_model=d_model), **overrides
    )


def llama_style_config(**overrides) -> GPTConfig:
    """The 2023+ default: RMSNorm, RoPE, SwiGLU, no biases, untied embeddings."""
    base = GPTConfig(
        norm="rmsnorm", position="rope", mlp="swiglu", bias=False, tie_embeddings=False
    )
    return replace(base, **overrides)


def nanochat_style_config(**overrides) -> GPTConfig:
    """The core of nanochat's GPT (2025-26): parameter-free RMSNorm, RoPE, ReLU^2, QK-norm,
    untied embeddings, soft-capped logits. (nanochat adds more tricks; see lesson 06 §2.6.)"""
    base = GPTConfig(
        norm="rmsnorm",
        norm_affine=False,
        position="rope",
        mlp="relu2",
        bias=False,
        tie_embeddings=False,
        qk_norm=True,
        logit_softcap=15.0,
    )
    return replace(base, **overrides)


# ---------------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """x / rms(x) * g. No mean subtraction, no bias (Zhang & Sennrich 2019)."""

    def __init__(self, dim: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y * self.weight if self.weight is not None else y


def make_norm(config: GPTConfig, dim: int | None = None) -> nn.Module:
    dim = dim or config.d_model
    if config.norm == "layernorm":
        return nn.LayerNorm(
            dim,
            eps=config.norm_eps,
            elementwise_affine=config.norm_affine,
            bias=config.bias,
        )
    if config.norm == "rmsnorm":
        return RMSNorm(dim, eps=config.norm_eps, affine=config.norm_affine)
    raise ValueError(f"unknown norm {config.norm!r}")


def rope_cache(
    length: int, head_dim: int, base: float = 10_000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of the rotation angles m * theta_i, shape (length, head_dim / 2)."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(torch.arange(length, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x_i, x_{i + d/2}) of ``x`` (B, H, T, D) by the angles of each position.

    This is the "half split" layout of GPT-NeoX and Llama; the RoFormer paper pairs adjacent
    dimensions instead. Both give the same relative-position property (lesson 04 §2.6).
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """softmax(q k^T / sqrt(d) + mask) v, written out. Shapes (B, H, T, D); used by tests and
    lesson 04 to check the fused kernel. Queries are aligned with the last keys."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if causal:
        t_q, t_k = q.shape[-2], k.shape[-2]
        allowed = torch.ones(t_q, t_k, dtype=torch.bool, device=q.device).tril(
            t_k - t_q
        )
        scores = scores.masked_fill(~allowed, float("-inf"))
    return scores.softmax(dim=-1) @ v


class KVCache:
    """Keys and values of every layer for the tokens seen so far (lesson 09).

    Stored after RoPE, so a cached key never has to be rotated again. Pre-allocated for
    ``max_length`` tokens; ``length`` is how many are filled.
    """

    def __init__(
        self,
        config: GPTConfig,
        batch_size: int,
        max_length: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (
            config.n_layer,
            batch_size,
            config.kv_heads,
            max_length,
            config.head_dim,
        )
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0
        self.max_length = max_length

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this step's k, v (B, Hkv, T, D) and return all keys and values so far."""
        end = self.length + k.shape[2]
        if end > self.max_length:
            raise ValueError(f"KV cache full: {end} > {self.max_length}")
        self.k[layer, :, :, self.length : end] = k
        self.v[layer, :, :, self.length : end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, num_tokens: int) -> None:
        self.length += num_tokens

    def num_bytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


class Attention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        hd = config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.n_head * hd, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.kv_heads * hd, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.kv_heads * hd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_head * hd, config.d_model, bias=config.bias)
        if config.qk_norm:
            self.q_norm = RMSNorm(hd, eps=config.norm_eps, affine=False)
            self.k_norm = RMSNorm(hd, eps=config.norm_eps, affine=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None,
        layer: int,
        kv_cache: KVCache | None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, cfg.n_head, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        if cfg.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)

        past = 0
        if kv_cache is not None:
            past = kv_cache.length
            k, v = kv_cache.update(layer, k, v)
        if cfg.kv_heads != cfg.n_head:
            # grouped-query attention: each group of query heads shares one key/value head
            repeat = cfg.n_head // cfg.kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        if padding_mask is not None:
            # (B, T) with True at real tokens -> (B, 1, 1, T) mask over keys
            mask = padding_mask[:, None, None, :]
            if cfg.causal:
                mask = mask & torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif not cfg.causal or T == 1:
            y = F.scaled_dot_product_attention(q, k, v)
        elif past == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # several new tokens after a cached prefix: token i sees keys 0 .. past + i
            allowed = torch.ones(T, past + T, dtype=torch.bool, device=x.device).tril(
                past
            )
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        y = y.transpose(1, 2).reshape(B, T, cfg.n_head * cfg.head_dim)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.kind = config.mlp
        self.up_proj = nn.Linear(config.d_model, config.ff_dim, bias=config.bias)
        if self.kind == "swiglu":
            self.gate_proj = nn.Linear(config.d_model, config.ff_dim, bias=config.bias)
        self.down_proj = nn.Linear(config.ff_dim, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "gelu":
            # GPT-2 used the tanh approximation of GELU
            h = F.gelu(self.up_proj(x), approximate="tanh")
        elif self.kind == "swiglu":
            h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        elif self.kind == "relu2":
            h = F.relu(self.up_proj(x)).square()
        else:
            raise ValueError(f"unknown mlp {self.kind!r}")
        return self.down_proj(h)


class Block(nn.Module):
    """Pre-norm residual block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm1 = make_norm(config)
        self.attn = Attention(config)
        self.norm2 = make_norm(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None,
        layer: int,
        kv_cache: KVCache | None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope, layer, kv_cache, padding_mask)
        return x + self.mlp(self.norm2(x))


# ---------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        if config.position == "learned":
            self.position_embedding = nn.Embedding(
                config.context_length, config.d_model
            )
        elif config.position == "rope":
            cos, sin = rope_cache(
                config.context_length, config.head_dim, config.rope_base
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            raise ValueError(f"unknown position {config.position!r}")
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.final_norm = make_norm(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """GPT-2 initialization: N(0, 0.02) everywhere, and the projections that write into the
        residual stream scaled down by 1/sqrt(2 * n_layer) (GPT-2 paper §2.3)."""
        residual_std = INIT_STD / math.sqrt(2 * self.config.n_layer)
        for name, p in self.named_parameters():
            if p.dim() < 2:
                if name.endswith("bias"):
                    nn.init.zeros_(p)
                else:
                    nn.init.ones_(p)
            elif name.endswith(("out_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, std=residual_std)
            else:
                nn.init.normal_(p, std=INIT_STD)

    def hidden_states(
        self,
        idx: torch.Tensor,
        kv_cache: KVCache | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Final normalized hidden states (B, T, d_model): what the LM head reads, and what an
        embedding model pools (lesson 13)."""
        B, T = idx.shape
        start = kv_cache.length if kv_cache is not None else 0
        if start + T > self.config.context_length:
            raise ValueError(
                f"sequence of {start + T} tokens exceeds the context length"
            )
        x = self.token_embedding(idx)
        rope = None
        if self.config.position == "learned":
            positions = torch.arange(start, start + T, device=idx.device)
            x = x + self.position_embedding(positions)
        else:
            rope = (self.rope_cos[start : start + T], self.rope_sin[start : start + T])
        for layer, block in enumerate(self.blocks):
            x = block(x, rope, layer, kv_cache, padding_mask)
        if kv_cache is not None:
            kv_cache.advance(T)
        return self.final_norm(x)

    def forward(
        self, idx: torch.Tensor, kv_cache: KVCache | None = None
    ) -> torch.Tensor:
        """Next-token logits (B, T, vocab_size) for token ids ``idx`` (B, T)."""
        logits = self.lm_head(self.hidden_states(idx, kv_cache))
        if self.config.logit_softcap is not None:
            cap = self.config.logit_softcap
            logits = cap * torch.tanh(logits / cap)
        return logits

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Parameter count (shared tensors counted once). ``exclude_embeddings`` drops the token
        and position embeddings, the convention of Kaplan et al. (2020)."""
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.token_embedding.weight.numel()
            if self.config.position == "learned":
                total -= self.position_embedding.weight.numel()
        return total

    def new_kv_cache(self, batch_size: int, max_length: int | None = None) -> KVCache:
        p = next(self.parameters())
        return KVCache(
            self.config,
            batch_size,
            max_length or self.config.context_length,
            p.device,
            p.dtype,
        )


# ---------------------------------------------------------------------------------------------
# Checkpoints and the original GPT-2 weights
# ---------------------------------------------------------------------------------------------


def save_model(model: GPT, path: Path, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": model.config.to_dict(), "state_dict": model.state_dict(), **extra},
        path,
    )


def load_model(
    path: Path, map_location: str | torch.device = "cpu"
) -> tuple[GPT, dict]:
    """Returns the model and the whole checkpoint dict (for any extra fields saved with it)."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


SAFETENSORS_DTYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}


def read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """A ``.safetensors`` file is an 8-byte header length, a JSON header, then raw bytes.

    The header maps each tensor name to its dtype, shape and byte range; nothing is executed
    while loading, which is the point of the format (unlike pickle-based ``.bin`` files).
    """
    raw = path.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len])
    data = memoryview(raw)[8 + header_len :]
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        dtype = SAFETENSORS_DTYPES[info["dtype"]]
        flat = torch.frombuffer(bytearray(data[start:end]), dtype=dtype)
        tensors[name] = flat.reshape(info["shape"])
    return tensors


def gpt2_from_state_dict(hf: dict[str, torch.Tensor], config: GPTConfig) -> GPT:
    """Map OpenAI's GPT-2 tensor names onto this module.

    GPT-2 stores its linear layers as ``Conv1D`` with weights of shape (in, out), the transpose
    of ``nn.Linear``, and fuses q, k, v into one ``c_attn`` matrix.
    """
    model = GPT(config)
    d = config.d_model
    ours = {
        "token_embedding.weight": hf["wte.weight"],
        "position_embedding.weight": hf["wpe.weight"],
        "final_norm.weight": hf["ln_f.weight"],
        "final_norm.bias": hf["ln_f.bias"],
    }
    for i in range(config.n_layer):
        p = f"h.{i}."
        q_w, k_w, v_w = hf[p + "attn.c_attn.weight"].split(d, dim=1)
        q_b, k_b, v_b = hf[p + "attn.c_attn.bias"].split(d, dim=0)
        ours |= {
            f"blocks.{i}.norm1.weight": hf[p + "ln_1.weight"],
            f"blocks.{i}.norm1.bias": hf[p + "ln_1.bias"],
            f"blocks.{i}.attn.q_proj.weight": q_w.T,
            f"blocks.{i}.attn.q_proj.bias": q_b,
            f"blocks.{i}.attn.k_proj.weight": k_w.T,
            f"blocks.{i}.attn.k_proj.bias": k_b,
            f"blocks.{i}.attn.v_proj.weight": v_w.T,
            f"blocks.{i}.attn.v_proj.bias": v_b,
            f"blocks.{i}.attn.out_proj.weight": hf[p + "attn.c_proj.weight"].T,
            f"blocks.{i}.attn.out_proj.bias": hf[p + "attn.c_proj.bias"],
            f"blocks.{i}.norm2.weight": hf[p + "ln_2.weight"],
            f"blocks.{i}.norm2.bias": hf[p + "ln_2.bias"],
            f"blocks.{i}.mlp.up_proj.weight": hf[p + "mlp.c_fc.weight"].T,
            f"blocks.{i}.mlp.up_proj.bias": hf[p + "mlp.c_fc.bias"],
            f"blocks.{i}.mlp.down_proj.weight": hf[p + "mlp.c_proj.weight"].T,
            f"blocks.{i}.mlp.down_proj.bias": hf[p + "mlp.c_proj.bias"],
        }
    # lm_head.weight is tied to the token embedding, so it is filled by the line above
    missing, unexpected = model.load_state_dict(
        {k: v.contiguous() for k, v in ours.items()}, strict=False
    )
    assert not unexpected and missing == ["lm_head.weight"], (missing, unexpected)
    return model


def load_gpt2_pretrained() -> GPT:
    """OpenAI's 124M GPT-2, downloaded once (548 MB) from the Hugging Face mirror."""
    from lm_course.data import gpt2_file

    weights = read_safetensors(gpt2_file("model.safetensors"))
    return gpt2_from_state_dict(weights, gpt2_config("gpt2"))


@dataclass
class ModelSummary:
    total: int
    non_embedding: int
    by_group: dict[str, int] = field(default_factory=dict)


def parameter_summary(model: GPT) -> ModelSummary:
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        key = (
            name.split(".")[0] if not name.startswith("blocks.") else name.split(".")[2]
        )
        groups[key] = groups.get(key, 0) + p.numel()
    return ModelSummary(
        model.num_parameters(), model.num_parameters(exclude_embeddings=True), groups
    )
