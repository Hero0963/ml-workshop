# scripts/pretrain.py
"""Pretrain the course's base model on TinyStories (the same run as lab 06, from the command
line). With a GPU, raise --steps and --batch-size for a much better model.

    uv run python scripts/pretrain.py --steps 1500
"""

import argparse
from pathlib import Path

import torch
from loguru import logger

from lm_course.artifacts import (
    BASE_MODEL_PATH,
    base_model_config,
    course_tokenizer,
    course_tokens,
)
from lm_course.model import GPT, save_model
from lm_course.optim import build_optimizer, wsd_schedule
from lm_course.training import evaluate, train
from lm_course.utils import count_parameters, get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--lr", type=float, default=3e-3, help="AdamW lr (embeddings, head)"
    )
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--out", type=Path, default=BASE_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    tokenizer = course_tokenizer()
    train_tokens, val_tokens = course_tokens(tokenizer)
    model = GPT(base_model_config()).to(device)
    logger.info(
        f"{count_parameters(model):,} parameters, {len(train_tokens):,} training tokens"
    )
    optimizer = build_optimizer(model, "muon", lr=args.lr, muon_lr=args.muon_lr)
    result = train(
        model,
        train_tokens,
        val_tokens,
        num_steps=args.steps,
        batch_size=args.batch_size,
        optimizer=optimizer,
        schedule=wsd_schedule,
        warmup=50,
        eval_every=250,
    )
    token_bytes = torch.tensor(tokenizer.token_byte_lengths())
    final = evaluate(
        model, val_tokens, model.config.context_length, 32, 40, token_bytes
    )
    logger.info(
        f"done in {result.seconds / 60:.1f} min: val loss {final.loss:.3f}, "
        f"{final.bits_per_byte:.3f} bits/byte"
    )
    save_model(model, args.out, val_loss=final.loss, bits_per_byte=final.bits_per_byte)


if __name__ == "__main__":
    main()
