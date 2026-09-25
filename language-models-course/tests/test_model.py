# tests/test_model.py
import json
import struct

import pytest
import torch

from helpers import download_or_skip
from lm_course.model import (
    GPT,
    GPTConfig,
    RMSNorm,
    apply_rope,
    attention_reference,
    gpt2_config,
    gpt2_from_state_dict,
    llama_style_config,
    load_gpt2_pretrained,
    load_model,
    nanochat_style_config,
    read_safetensors,
    rope_cache,
    save_model,
)

SMALL = dict(vocab_size=97, context_length=32, n_layer=2, n_head=4, d_model=32)
PRESETS = {
    "gpt2": gpt2_config("gpt2", **SMALL),
    "llama_gqa": llama_style_config(**SMALL, n_kv_head=2),
    "nanochat": nanochat_style_config(**SMALL),
}


@pytest.mark.parametrize(
    "size, expected",
    [("gpt2", 124_439_808), ("gpt2-medium", 354_823_168)],
)
def test_gpt2_parameter_counts(size: str, expected: int) -> None:
    with torch.device("meta"):
        model = GPT(gpt2_config(size))
    assert model.num_parameters() == expected


@pytest.mark.parametrize("name", list(PRESETS))
def test_causal_logits_ignore_future_tokens(name: str) -> None:
    model = GPT(PRESETS[name]).eval()
    x = torch.randint(0, 97, (2, 16))
    y = x.clone()
    y[:, 10:] = torch.randint(0, 97, (2, 6))
    with torch.no_grad():
        a, b = model(x), model(y)
    torch.testing.assert_close(a[:, :10], b[:, :10])
    assert not torch.allclose(a[:, 10:], b[:, 10:])


@pytest.mark.parametrize("name", list(PRESETS))
def test_kv_cache_matches_full_forward(name: str) -> None:
    model = GPT(PRESETS[name]).eval()
    x = torch.randint(0, 97, (3, 20))
    with torch.no_grad():
        full = model(x)
        cache = model.new_kv_cache(3, 20)
        steps = [model(x[:, :8], kv_cache=cache)]  # prefill
        steps += [model(x[:, 8:12], kv_cache=cache)]  # a chunk after the prefix
        steps += [
            model(x[:, t : t + 1], kv_cache=cache) for t in range(12, 20)
        ]  # decode
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-4)
    assert cache.length == 20


def test_attention_module_matches_the_written_out_formula() -> None:
    q, k, v = torch.randn(3, 2, 4, 7, 16).unbind(0)
    expected = attention_reference(q, k, v, causal=True)
    got = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
    # the first query can only see the first key
    torch.testing.assert_close(expected[:, :, 0], v[:, :, 0])


def test_rope_scores_depend_only_on_relative_position() -> None:
    cos, sin = rope_cache(64, 16)
    q, k = torch.randn(2, 16)

    def score(m: int, n: int) -> torch.Tensor:
        qm = apply_rope(q.view(1, 1, 1, 16), cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k.view(1, 1, 1, 16), cos[n : n + 1], sin[n : n + 1])
        return (qm * kn).sum()

    torch.testing.assert_close(score(5, 2), score(40, 37), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(score(0, 0), (q * k).sum(), atol=1e-5, rtol=1e-5)
    assert not torch.allclose(score(5, 2), score(5, 3))


def test_rope_is_a_rotation() -> None:
    cos, sin = rope_cache(32, 8)
    x = torch.randn(1, 1, 32, 8)
    torch.testing.assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1))


def test_rmsnorm_output_has_unit_rms() -> None:
    y = RMSNorm(64, affine=False)(3 * torch.randn(5, 64) + 2)
    torch.testing.assert_close(
        y.pow(2).mean(dim=-1), torch.ones(5), atol=1e-4, rtol=1e-4
    )


def test_grouped_query_attention_shrinks_the_kv_projections() -> None:
    mha = GPT(llama_style_config(**SMALL))
    gqa = GPT(llama_style_config(**SMALL, n_kv_head=1))
    saved = (
        2
        * SMALL["n_layer"]
        * SMALL["d_model"]
        * (SMALL["d_model"] - SMALL["d_model"] // 4)
    )
    assert mha.num_parameters() - gqa.num_parameters() == saved
    assert gqa.new_kv_cache(1).num_bytes() * 4 == mha.new_kv_cache(1).num_bytes()


def test_encoder_mode_sees_both_directions_and_ignores_padding() -> None:
    model = GPT(llama_style_config(**SMALL, causal=False)).eval()
    x = torch.randint(0, 97, (1, 12))
    y = x.clone()
    y[:, -1] = (y[:, -1] + 1) % 97
    with torch.no_grad():
        assert not torch.allclose(
            model.hidden_states(x)[:, 0], model.hidden_states(y)[:, 0]
        )
        mask = torch.ones(1, 12, dtype=torch.bool)
        mask[:, 9:] = False  # the last three tokens are padding
        a = model.hidden_states(x, padding_mask=mask)[:, :9]
        b = model.hidden_states(y, padding_mask=mask)[:, :9]
    torch.testing.assert_close(a, b)


def test_tied_embeddings_share_one_tensor() -> None:
    tied = GPT(gpt2_config("gpt2", **SMALL))
    untied = GPT(llama_style_config(**SMALL))
    assert tied.lm_head.weight is tied.token_embedding.weight
    assert untied.lm_head.weight is not untied.token_embedding.weight


def test_residual_projections_start_smaller() -> None:
    model = GPT(
        gpt2_config(
            "gpt2", vocab_size=97, context_length=32, n_layer=8, n_head=4, d_model=256
        )
    )
    std_in = model.blocks[0].mlp.up_proj.weight.std().item()
    std_out = model.blocks[0].mlp.down_proj.weight.std().item()
    assert std_in == pytest.approx(0.02, rel=0.05)
    assert std_out == pytest.approx(0.02 / 4, rel=0.05)  # 1 / sqrt(2 * 8)


def test_save_and_load_round_trip(tmp_path) -> None:  # noqa: ANN001
    model = GPT(PRESETS["llama_gqa"]).eval()
    save_model(model, tmp_path / "m.pt", note="hello")
    loaded, checkpoint = load_model(tmp_path / "m.pt")
    x = torch.randint(0, 97, (1, 10))
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x))
    assert checkpoint["note"] == "hello"
    assert loaded.config == model.config


def test_read_safetensors(tmp_path) -> None:  # noqa: ANN001
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.tensor([1.5, -2.0], dtype=torch.float32)
    blob_a, blob_b = a.numpy().tobytes(), b.numpy().tobytes()
    header = {
        "a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, len(blob_a)]},
        "b": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [len(blob_a), len(blob_a) + 8],
        },
        "__metadata__": {"format": "pt"},
    }
    raw = json.dumps(header).encode()
    (tmp_path / "t.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw + blob_a + blob_b
    )
    tensors = read_safetensors(tmp_path / "t.safetensors")
    torch.testing.assert_close(tensors["a"], a)
    torch.testing.assert_close(tensors["b"], b)


def _export_to_gpt2_names(model: GPT) -> dict[str, torch.Tensor]:
    """The inverse of gpt2_from_state_dict: our weights under OpenAI's Conv1D names."""
    sd = model.state_dict()
    out = {
        "wte.weight": sd["token_embedding.weight"],
        "wpe.weight": sd["position_embedding.weight"],
        "ln_f.weight": sd["final_norm.weight"],
        "ln_f.bias": sd["final_norm.bias"],
    }
    for i in range(model.config.n_layer):
        b = f"blocks.{i}."
        qkv_w = [sd[b + f"attn.{n}_proj.weight"].T for n in "qkv"]
        qkv_b = [sd[b + f"attn.{n}_proj.bias"] for n in "qkv"]
        out |= {
            f"h.{i}.ln_1.weight": sd[b + "norm1.weight"],
            f"h.{i}.ln_1.bias": sd[b + "norm1.bias"],
            f"h.{i}.attn.c_attn.weight": torch.cat(qkv_w, dim=1),
            f"h.{i}.attn.c_attn.bias": torch.cat(qkv_b),
            f"h.{i}.attn.c_proj.weight": sd[b + "attn.out_proj.weight"].T,
            f"h.{i}.attn.c_proj.bias": sd[b + "attn.out_proj.bias"],
            f"h.{i}.ln_2.weight": sd[b + "norm2.weight"],
            f"h.{i}.ln_2.bias": sd[b + "norm2.bias"],
            f"h.{i}.mlp.c_fc.weight": sd[b + "mlp.up_proj.weight"].T,
            f"h.{i}.mlp.c_fc.bias": sd[b + "mlp.up_proj.bias"],
            f"h.{i}.mlp.c_proj.weight": sd[b + "mlp.down_proj.weight"].T,
            f"h.{i}.mlp.c_proj.bias": sd[b + "mlp.down_proj.bias"],
        }
    return out


def test_gpt2_weight_mapping_round_trips() -> None:
    config = PRESETS["gpt2"]
    model = GPT(config).eval()
    reloaded = gpt2_from_state_dict(_export_to_gpt2_names(model), config).eval()
    x = torch.randint(0, 97, (2, 12))
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x), model(x))


def test_config_rejects_bad_head_counts() -> None:
    with pytest.raises(ValueError):
        GPTConfig(d_model=30, n_head=4)
    with pytest.raises(ValueError):
        GPTConfig(d_model=32, n_head=4, n_kv_head=3)


@pytest.mark.network
def test_pretrained_gpt2_matches_reference_logits() -> None:
    model = download_or_skip(load_gpt2_pretrained).eval()
    # reference: Hugging Face transformers GPT2LMHeadModel("openai-community/gpt2"), 2026-09-25
    cases = {
        (464, 3139, 286, 4881, 318): ([262, 783, 257, 4881, 6342], -100.25),
        (15496, 11, 616, 1438, 318): ([1757, 509, 449, 3899, 406], -64.443),
    }
    for prompt, (top5, top_logit) in cases.items():
        with torch.no_grad():
            logits = model(torch.tensor([prompt]))[0, -1]
        assert torch.topk(logits, 5).indices.tolist() == top5
        assert logits.max().item() == pytest.approx(top_logit, abs=1e-2)
