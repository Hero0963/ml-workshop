# scripts/train_nanogpt.py
"""
Train a character-level GPT on Shakespeare text.
Simplified version of nanoGPT's train.py for educational purposes.

Usage:
    uv run python scripts/train_nanogpt.py
    uv run python scripts/train_nanogpt.py --device=cpu --max_iters=500
"""

import os
import math
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Model Definition (simplified nanoGPT)
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        # special scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_shakespeare(project_dir):
    """Load Shakespeare text from references or download."""
    paths = [
        os.path.join(
            project_dir,
            "references",
            "nanoGPT",
            "data",
            "shakespeare_char",
            "input.txt",
        ),
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()

    print("Shakespeare text not found. Downloading...")
    import urllib.request

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_dir = os.path.join(project_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "input.txt")
    urllib.request.urlretrieve(url, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train nanoGPT on Shakespeare")
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu/mps)"
    )
    parser.add_argument(
        "--max_iters", type=int, default=3000, help="Training iterations"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=250, help="Eval every N steps"
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--block_size", type=int, default=None, help="Context length")
    parser.add_argument("--n_layer", type=int, default=None, help="Number of layers")
    parser.add_argument("--n_head", type=int, default=None, help="Number of heads")
    parser.add_argument("--n_embd", type=int, default=None, help="Embedding dimension")
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Adjust defaults based on device
    if device == "cpu":
        default_block_size = 64
        default_batch_size = 12
        default_n_layer = 4
        default_n_head = 4
        default_n_embd = 128
        default_dropout = 0.0
    else:
        default_block_size = 256
        default_batch_size = 64
        default_n_layer = 6
        default_n_head = 6
        default_n_embd = 384
        default_dropout = 0.2

    block_size = args.block_size or default_block_size
    batch_size = args.batch_size or default_batch_size
    n_layer = args.n_layer or default_n_layer
    n_head = args.n_head or default_n_head
    n_embd = args.n_embd or default_n_embd

    print("=" * 60)
    print("nanoGPT Training - Shakespeare Character-Level")
    print("=" * 60)
    print(f"Device:     {device}")
    print(f"Block size: {block_size}")
    print(f"Batch size: {batch_size}")
    print(f"Layers:     {n_layer}")
    print(f"Heads:      {n_head}")
    print(f"Embedding:  {n_embd}")
    print(f"Max iters:  {args.max_iters}")
    print()

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    text = load_shakespeare(project_dir)
    print(f"Loaded {len(text):,} characters of Shakespeare")

    # Character-level tokenization
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join([itos[i] for i in ids])

    print(f"Vocab size: {vocab_size}")

    # Prepare data
    data = np.array(encode(text), dtype=np.int64)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = np.random.randint(0, len(d) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(d[i : i + block_size].copy()) for i in ix])
        y = torch.stack(
            [torch.from_numpy(d[i + 1 : i + 1 + block_size].copy()) for i in ix]
        )
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model, eval_iters=20):
        model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = []
            for _ in range(eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    # Create model
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=default_dropout,
    )
    model = GPT(config).to(device)
    n_params = model.get_num_params()
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    # Training loop
    print("\n--- Training ---")
    t0 = time.time()

    for iter_num in range(args.max_iters):
        # Evaluate
        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
            losses = estimate_loss(model)
            elapsed = time.time() - t0
            print(
                f"Step {iter_num:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"time {elapsed:.1f}s"
            )

        # Training step
        X, Y = get_batch("train")
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.time() - t0
    print(f"\nTraining completed in {total_time:.1f}s")

    # Save model
    out_dir = os.path.join(project_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "shakespeare_char.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "chars": chars,
        },
        ckpt_path,
    )
    print(f"Model saved to {ckpt_path}")

    # Generate sample
    print("\n--- Generated Text Sample ---")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)
    print(decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
