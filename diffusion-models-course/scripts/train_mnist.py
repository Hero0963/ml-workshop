# scripts/train_mnist.py
"""Train an image diffusion model for longer than the lab notebooks do (a GPU helps).

Examples:
    uv run python scripts/train_mnist.py --method ddpm --arch unet --steps 20000
    uv run python scripts/train_mnist.py --method flow --arch dit --conditional --steps 20000
"""

import argparse
from pathlib import Path

import torch
from loguru import logger
from torch import nn

from diffusion_course.data import (
    IMAGE_DATASETS,
    NUM_IMAGE_CLASSES,
    image_loader,
    infinite_batches,
)
from diffusion_course.ddpm import ddpm_loss
from diffusion_course.flow_matching import flow_matching_loss
from diffusion_course.guidance import drop_labels
from diffusion_course.models import build_model
from diffusion_course.schedules import NoiseSchedule
from diffusion_course.training import save_checkpoint, train
from diffusion_course.utils import (
    CHECKPOINT_DIR,
    count_parameters,
    get_device,
    set_seed,
)

LABEL_DROP_PROB = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=list(IMAGE_DATASETS), default="mnist")
    parser.add_argument("--method", choices=["ddpm", "flow"], default="ddpm")
    parser.add_argument("--arch", choices=["unet", "dit"], default="unet")
    parser.add_argument("--conditional", action="store_true", help="train for CFG")
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--base-channels", type=int, default=32, help="U-Net width")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    num_classes = NUM_IMAGE_CLASSES if args.conditional else None
    model_kwargs: dict = {"num_classes": num_classes}
    if args.arch == "unet":
        model_kwargs["base_channels"] = args.base_channels
    model = build_model(args.arch, **model_kwargs)
    logger.info(
        f"{args.arch}: {count_parameters(model) / 1e6:.2f}M parameters on {device}"
    )

    schedule = NoiseSchedule.create(args.schedule).to(device)

    def loss_fn(m: nn.Module, x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        y = drop_labels(y, LABEL_DROP_PROB, NUM_IMAGE_CLASSES) if num_classes else None
        if args.method == "ddpm":
            return ddpm_loss(m, schedule, x, y)
        return flow_matching_loss(m, x, y)

    batches = infinite_batches(image_loader(args.dataset, args.batch_size))
    result = train(
        model,
        loss_fn,
        batches,
        args.steps,
        lr=args.lr,
        ema_decay=args.ema_decay,
        device=device,
    )
    logger.info(f"trained {args.steps} steps in {result.seconds / 60:.1f} min")

    suffix = "_cond" if args.conditional else ""
    out = args.out or (
        CHECKPOINT_DIR / f"{args.dataset}_{args.method}_{args.arch}{suffix}.pt"
    )
    save_checkpoint(
        out,
        result.sampling_model,
        args.arch,
        model_kwargs,
        method=args.method,
        dataset=args.dataset,
        schedule=args.schedule,
        num_classes=num_classes,
    )


if __name__ == "__main__":
    main()
