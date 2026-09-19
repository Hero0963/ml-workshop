# src/core/tests/vl_models/test_merge_lora.py
"""The merger reimplements peft's arithmetic, so these tests are what stand in for it.

Every case builds a two-tensor checkpoint on disk and merges it, which runs the real
shard streaming and the real file layout at a size that costs milliseconds. The point of
the refusals is that a wrong merge is silent: the model still loads and still answers,
it is just no longer the model that scored 200/200.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from src.core.vl_models.merge_lora import (
    ADAPTER_CONFIG_FILENAME,
    ADAPTER_WEIGHTS_FILENAME,
    WEIGHT_INDEX_FILENAME,
    MergeError,
    copy_support_files,
    collect_lora_pairs,
    load_adapter_config,
    merge,
    merged_weight,
)

SHARD_NAME = "model.safetensors-00001-of-00001.safetensors"
TARGET = "model.visual.blocks.0.attn.proj.weight"
UNTOUCHED = "model.language_model.embed_tokens.weight"
RANK = 2
OUT_FEATURES = 4
IN_FEATURES = 3
CHAT_TEMPLATE = "chat_template.jinja"
CHAT_TEMPLATE_TEXT = "{{ messages }}"

BASE_ADAPTER_CONFIG = {
    "r": RANK,
    "lora_alpha": RANK,
    "use_dora": False,
    "use_rslora": False,
    "fan_in_fan_out": False,
    "lora_bias": False,
    "modules_to_save": None,
    "rank_pattern": {},
    "alpha_pattern": {},
}


def _write_base(base_dir: Path, base_weight: torch.Tensor) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    untouched = torch.ones(2, 2, dtype=torch.bfloat16)
    save_file({TARGET: base_weight, UNTOUCHED: untouched}, base_dir / SHARD_NAME)
    index = {"weight_map": {TARGET: SHARD_NAME, UNTOUCHED: SHARD_NAME}}
    (base_dir / WEIGHT_INDEX_FILENAME).write_text(json.dumps(index), encoding="utf-8")
    (base_dir / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    (base_dir / "tokenizer_config.json").write_text(
        '{"tokenizer_class": "Qwen2Tokenizer"}', encoding="utf-8"
    )
    (base_dir / CHAT_TEMPLATE).write_text(CHAT_TEMPLATE_TEXT, encoding="utf-8")


def _write_adapter(
    adapter_dir: Path,
    a_matrix: torch.Tensor,
    b_matrix: torch.Tensor,
    config: dict | None = None,
    target: str = TARGET,
    chat_template: str = CHAT_TEMPLATE_TEXT,
) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    stem = "base_model.model." + target.removesuffix(".weight")
    save_file(
        {f"{stem}.lora_A.weight": a_matrix, f"{stem}.lora_B.weight": b_matrix},
        adapter_dir / ADAPTER_WEIGHTS_FILENAME,
    )
    (adapter_dir / ADAPTER_CONFIG_FILENAME).write_text(
        json.dumps(config or BASE_ADAPTER_CONFIG), encoding="utf-8"
    )
    (adapter_dir / "tokenizer_config.json").write_text(
        '{"tokenizer_class": "TokenizersBackend"}', encoding="utf-8"
    )
    (adapter_dir / CHAT_TEMPLATE).write_text(chat_template, encoding="utf-8")


@pytest.fixture
def lora_pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    a_matrix = torch.randn(RANK, IN_FEATURES, dtype=torch.float32)
    b_matrix = torch.randn(OUT_FEATURES, RANK, dtype=torch.float32)
    return a_matrix, b_matrix


class TestMergedWeight:
    def test_applies_b_times_a_scaled(self, lora_pair):
        a_matrix, b_matrix = lora_pair
        base = torch.zeros(OUT_FEATURES, IN_FEATURES, dtype=torch.float32)

        result = merged_weight(base, a_matrix, b_matrix, scaling=2.0)

        assert torch.allclose(result, 2.0 * (b_matrix @ a_matrix))

    def test_keeps_the_base_dtype(self, lora_pair):
        """The checkpoint is bfloat16; a merge that silently widens it doubles its size."""
        a_matrix, b_matrix = lora_pair
        base = torch.zeros(OUT_FEATURES, IN_FEATURES, dtype=torch.bfloat16)

        assert (
            merged_weight(base, a_matrix, b_matrix, scaling=1.0).dtype == torch.bfloat16
        )

    def test_rejects_a_delta_of_the_wrong_shape(self, lora_pair):
        a_matrix, b_matrix = lora_pair
        base = torch.zeros(OUT_FEATURES + 1, IN_FEATURES, dtype=torch.float32)

        with pytest.raises(MergeError, match="does not fit"):
            merged_weight(base, a_matrix, b_matrix, scaling=1.0)


class TestLoadAdapterConfig:
    @pytest.mark.parametrize(
        "key", ["use_dora", "use_rslora", "fan_in_fan_out", "lora_bias"]
    )
    def test_refuses_variants_it_does_not_implement(self, tmp_path, lora_pair, key):
        _write_adapter(tmp_path, *lora_pair, config={**BASE_ADAPTER_CONFIG, key: True})

        with pytest.raises(MergeError, match=key):
            load_adapter_config(tmp_path)

    def test_refuses_modules_to_save(self, tmp_path, lora_pair):
        """Those tensors are replaced wholesale, not added to, so ignoring them is wrong."""
        _write_adapter(
            tmp_path,
            *lora_pair,
            config={**BASE_ADAPTER_CONFIG, "modules_to_save": ["head"]},
        )

        with pytest.raises(MergeError, match="modules_to_save"):
            load_adapter_config(tmp_path)


class TestCollectLoraPairs:
    def test_names_pairs_after_the_base_tensor(self, tmp_path, lora_pair):
        _write_adapter(tmp_path, *lora_pair)

        assert list(collect_lora_pairs(tmp_path)) == [TARGET]

    def test_rejects_a_lora_a_with_no_lora_b(self, tmp_path, lora_pair):
        a_matrix, _ = lora_pair
        tmp_path.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                f"base_model.model.{TARGET.removesuffix('.weight')}.lora_A.weight": a_matrix
            },
            tmp_path / ADAPTER_WEIGHTS_FILENAME,
        )
        (tmp_path / ADAPTER_CONFIG_FILENAME).write_text(json.dumps(BASE_ADAPTER_CONFIG))

        with pytest.raises(MergeError, match="no partner"):
            collect_lora_pairs(tmp_path)


class TestMerge:
    def test_writes_a_checkpoint_with_the_delta_applied(self, tmp_path, lora_pair):
        a_matrix, b_matrix = lora_pair
        base_weight = torch.zeros(OUT_FEATURES, IN_FEATURES, dtype=torch.bfloat16)
        _write_base(tmp_path / "base", base_weight)
        _write_adapter(tmp_path / "adapter", a_matrix, b_matrix)

        merged_count = merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "out")

        assert merged_count == 1
        merged = load_file(tmp_path / "out" / SHARD_NAME)
        expected = (b_matrix @ a_matrix).to(torch.bfloat16)
        assert torch.allclose(
            merged[TARGET].to(torch.float32), expected.to(torch.float32)
        )

    def test_leaves_tensors_the_adapter_never_touched_alone(self, tmp_path, lora_pair):
        _write_base(
            tmp_path / "base",
            torch.zeros(OUT_FEATURES, IN_FEATURES, dtype=torch.bfloat16),
        )
        _write_adapter(tmp_path / "adapter", *lora_pair)

        merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "out")

        assert torch.equal(
            load_file(tmp_path / "out" / SHARD_NAME)[UNTOUCHED],
            torch.ones(2, 2, dtype=torch.bfloat16),
        )

    def test_copies_config_and_tokenizer_next_to_the_weights(self, tmp_path, lora_pair):
        """Ollama refuses a directory without config.json, so this is not cosmetic."""
        _write_base(
            tmp_path / "base",
            torch.zeros(OUT_FEATURES, IN_FEATURES, dtype=torch.bfloat16),
        )
        _write_adapter(tmp_path / "adapter", *lora_pair)

        merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "out")

        written = {path.name for path in (tmp_path / "out").iterdir()}
        assert {
            "config.json",
            WEIGHT_INDEX_FILENAME,
            "tokenizer_config.json",
        } <= written
        assert ADAPTER_CONFIG_FILENAME not in written

    def test_keeps_the_base_tokenizer_config_not_the_adapter_re_serialisation(
        self, tmp_path, lora_pair
    ):
        """The trainer rewrites it with transformers 5.x, naming a class 4.x cannot load."""
        _write_base(
            tmp_path / "base",
            torch.zeros(OUT_FEATURES, IN_FEATURES, dtype=torch.bfloat16),
        )
        _write_adapter(tmp_path / "adapter", *lora_pair)

        merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "out")

        written = json.loads(
            (tmp_path / "out" / "tokenizer_config.json").read_text("utf-8")
        )
        assert written["tokenizer_class"] == "Qwen2Tokenizer"

    def test_refuses_a_chat_template_that_drifted_from_the_base(
        self, tmp_path, lora_pair
    ):
        """Serving a template training never used is the defect that cost two rounds."""
        _write_base(
            tmp_path / "base",
            torch.zeros(OUT_FEATURES, IN_FEATURES, dtype=torch.bfloat16),
        )
        _write_adapter(
            tmp_path / "adapter", *lora_pair, chat_template="{{ something_else }}"
        )

        with pytest.raises(MergeError, match="chat_template.jinja differs"):
            copy_support_files(tmp_path / "base", tmp_path / "adapter", tmp_path)

    def test_refuses_an_adapter_built_for_a_different_base(self, tmp_path, lora_pair):
        """The wrong pairing merges cleanly and produces a quietly broken model."""
        _write_base(
            tmp_path / "base",
            torch.zeros(OUT_FEATURES, IN_FEATURES, dtype=torch.bfloat16),
        )
        _write_adapter(
            tmp_path / "adapter", *lora_pair, target="model.somewhere.else.weight"
        )

        with pytest.raises(MergeError, match="does not belong to it"):
            merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "out")
