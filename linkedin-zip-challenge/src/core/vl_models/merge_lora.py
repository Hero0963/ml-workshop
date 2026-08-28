# src/core/vl_models/merge_lora.py
"""Merges the P4c LoRA adapter into its base model, producing a plain checkpoint.

Usage:
    uv run python -m src.core.vl_models.merge_lora \
        --base models/base/Qwen3.5-4B \
        --adapter models/colab_finetune/p4c_qwen35_4b_zip_lora \
        --out models/merged/qwen35-4b-zip-p4c

Why this exists rather than ``peft.PeftModel.merge_and_unload()``: the adapter's base
is ``Qwen3.5-4B``, whose modelling code landed in ``transformers`` 5.x, while this
project pins ``transformers<5`` for the rest of the stack. Importing the model class
just to add two matrices per module would force that upgrade on every other module in
the repo. The arithmetic does not need it -- the tensors are read and written directly.

That shortcut is only safe because this particular adapter is the plain case, and the
loader refuses anything else rather than merging it wrongly:

*   ``use_dora`` / ``use_rslora`` / ``fan_in_fan_out`` are all off, so the update is
    exactly ``W += (alpha / r) * B @ A``.
*   ``modules_to_save`` and ``lora_bias`` are null, so no tensor outside the LoRA pairs
    changes and no bias term is involved.
*   ``rank_pattern`` / ``alpha_pattern`` are empty, so one scaling factor covers every
    module.

The merged output is written in the base model's own dtype. The delta is accumulated in
float32 first: the adapter is stored as float32 and ``B @ A`` in bfloat16 would throw
away most of what was trained, given the deltas are around 1e-1 against weights of a
similar magnitude.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"
WEIGHT_INDEX_FILENAME = "model.safetensors.index.json"

# peft prefixes every module it wrapped; the base checkpoint has no such prefix.
PEFT_PREFIX = "base_model.model."
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"

# Config keys whose only supported value is the one this merger implements.
UNSUPPORTED_WHEN_TRUTHY = ("use_dora", "use_rslora", "fan_in_fan_out", "lora_bias")
UNSUPPORTED_WHEN_NON_EMPTY = ("modules_to_save", "rank_pattern", "alpha_pattern")

# Everything the base ships that is not a weight shard: tokenizer, processor, config.
NON_WEIGHT_SUFFIXES = (".json", ".jinja", ".txt")

# Files that decide what the model actually sees. If the adapter's copies differ from
# the base's, one of the two renders prompts unlike training did.
BEHAVIOUR_DEFINING_FILES = (
    "chat_template.jinja",
    "tokenizer.json",
    "processor_config.json",
)


class MergeError(Exception):
    """The adapter and the base cannot be merged as they stand."""


def load_adapter_config(adapter_dir: Path) -> dict:
    """Reads the adapter config and refuses every variant this merger cannot do."""
    config = json.loads((adapter_dir / ADAPTER_CONFIG_FILENAME).read_text("utf-8"))

    for key in UNSUPPORTED_WHEN_TRUTHY:
        if config.get(key):
            raise MergeError(
                f"adapter_config.json has {key}={config[key]!r}. This merger only "
                "implements the plain LoRA update; use peft for anything else."
            )
    for key in UNSUPPORTED_WHEN_NON_EMPTY:
        if config.get(key):
            raise MergeError(
                f"adapter_config.json has a non-empty {key}={config[key]!r}, which "
                "changes tensors this merger does not touch."
            )
    return config


def collect_lora_pairs(
    adapter_dir: Path,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Maps each base tensor name to its ``(A, B)`` pair from the adapter file."""
    a_matrices: dict[str, torch.Tensor] = {}
    b_matrices: dict[str, torch.Tensor] = {}

    with safe_open(adapter_dir / ADAPTER_WEIGHTS_FILENAME, framework="pt") as handle:
        for key in handle.keys():
            if key.startswith(PEFT_PREFIX):
                stripped = key[len(PEFT_PREFIX) :]
            else:
                raise MergeError(
                    f"Adapter tensor {key!r} lacks the {PEFT_PREFIX!r} prefix."
                )
            if stripped.endswith(LORA_A_SUFFIX):
                a_matrices[stripped[: -len(LORA_A_SUFFIX)] + ".weight"] = (
                    handle.get_tensor(key)
                )
            elif stripped.endswith(LORA_B_SUFFIX):
                b_matrices[stripped[: -len(LORA_B_SUFFIX)] + ".weight"] = (
                    handle.get_tensor(key)
                )
            else:
                raise MergeError(
                    f"Adapter tensor {key!r} is neither a lora_A nor a lora_B."
                )

    unpaired = set(a_matrices) ^ set(b_matrices)
    if unpaired:
        raise MergeError(
            f"{len(unpaired)} LoRA matrices have no partner: {sorted(unpaired)[:3]}"
        )

    return {name: (a_matrices[name], b_matrices[name]) for name in a_matrices}


def merged_weight(
    base: torch.Tensor, a_matrix: torch.Tensor, b_matrix: torch.Tensor, scaling: float
) -> torch.Tensor:
    """Applies ``W += scaling * B @ A``, accumulating in float32."""
    delta = (b_matrix.to(torch.float32) @ a_matrix.to(torch.float32)) * scaling
    if delta.shape != base.shape:
        raise MergeError(
            f"LoRA delta {tuple(delta.shape)} does not fit {tuple(base.shape)}."
        )
    return (base.to(torch.float32) + delta).to(base.dtype)


def copy_support_files(base_dir: Path, adapter_dir: Path, out_dir: Path) -> None:
    """Copies the base's config and tokenizer files, after checking the adapter agrees.

    The tempting rule -- let the adapter's copies win, since it carries what training
    actually used -- is wrong here, and quietly so. The trainer re-serialises those files
    with whatever ``transformers`` it ran on, which was 5.x on Colab while this project
    pins 4.x; the resulting ``tokenizer_config.json`` names a tokenizer class that no 4.x
    tool can load, and GGUF conversion fails on it.

    So the base's copies are used, and the files that would actually change the model's
    behaviour are compared byte for byte instead. A mismatch there is the
    train/inference rendering trap from handover-vlm-parser.md section 7, which has cost
    this project a full evaluation round twice. It stops the merge rather than being
    resolved by a rule of thumb.
    """
    for path in sorted(base_dir.iterdir()):
        if not path.is_file() or path.suffix not in NON_WEIGHT_SUFFIXES:
            continue
        if path.name in (ADAPTER_CONFIG_FILENAME, WEIGHT_INDEX_FILENAME):
            continue
        shutil.copy2(path, out_dir / path.name)
        logger.debug("copied {}", path.name)

    for filename in BEHAVIOUR_DEFINING_FILES:
        adapter_copy = adapter_dir / filename
        base_copy = base_dir / filename
        if not adapter_copy.is_file() or not base_copy.is_file():
            continue
        if adapter_copy.read_bytes() != base_copy.read_bytes():
            raise MergeError(
                f"{filename} differs between the base and the adapter. Training used the "
                "adapter's copy, so serving the base's would render prompts differently "
                "from training -- decide which is right rather than letting this pass."
            )
        logger.debug("{} matches the base byte for byte", filename)


def merge(base_dir: Path, adapter_dir: Path, out_dir: Path) -> int:
    """Writes the merged checkpoint and returns how many tensors were changed."""
    config = load_adapter_config(adapter_dir)
    scaling = config["lora_alpha"] / config["r"]
    pairs = collect_lora_pairs(adapter_dir)
    logger.info(
        "{} LoRA pairs, scaling = {}/{} = {}",
        len(pairs),
        config["lora_alpha"],
        config["r"],
        scaling,
    )

    index = json.loads((base_dir / WEIGHT_INDEX_FILENAME).read_text("utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    missing = set(pairs) - set(weight_map)
    if missing:
        raise MergeError(
            f"{len(missing)} adapter targets are absent from the base checkpoint, so the "
            f"adapter does not belong to it. First few: {sorted(missing)[:3]}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_count = 0

    for shard_name in sorted(set(weight_map.values())):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(base_dir / shard_name, framework="pt") as handle:
            metadata = handle.metadata()
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if key in pairs:
                    a_matrix, b_matrix = pairs[key]
                    tensor = merged_weight(tensor, a_matrix, b_matrix, scaling)
                    merged_count += 1
                tensors[key] = tensor
        save_file(tensors, out_dir / shard_name, metadata=metadata)
        logger.info("wrote {} ({} tensors)", shard_name, len(tensors))
        del tensors

    shutil.copy2(base_dir / WEIGHT_INDEX_FILENAME, out_dir / WEIGHT_INDEX_FILENAME)
    copy_support_files(base_dir, adapter_dir, out_dir)

    if merged_count != len(pairs):
        raise MergeError(
            f"Merged {merged_count} tensors but the adapter holds {len(pairs)}."
        )
    return merged_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, required=True, help="Base checkpoint directory."
    )
    parser.add_argument(
        "--adapter", type=Path, required=True, help="LoRA adapter directory."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Where to write the merge."
    )
    args = parser.parse_args()

    merged_count = merge(args.base, args.adapter, args.out)
    logger.info("merged {} tensors into {}", merged_count, args.out)


if __name__ == "__main__":
    main()
