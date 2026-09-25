# scripts/sample_mnist.py
"""Draw a grid of samples from a checkpoint written by train_mnist.py.

Examples:
    uv run python scripts/sample_mnist.py checkpoints/mnist_ddpm_unet.pt --steps 50
    uv run python scripts/sample_mnist.py checkpoints/mnist_flow_dit_cond.pt --guidance-scale 3
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from loguru import logger

from diffusion_course.ddim import ddim_sample
from diffusion_course.flow_matching import flow_sample
from diffusion_course.guidance import ClassifierFreeGuidance
from diffusion_course.schedules import NoiseSchedule
from diffusion_course.training import load_checkpoint
from diffusion_course.utils import get_device, set_seed
from diffusion_course.viz import new_axes, show_images

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_PER_CLASS = 8
UNCONDITIONAL_SAMPLES = 64
IMAGE_SHAPE = (1, 28, 28)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=50, help="sampling steps")
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    plt.switch_backend("Agg")
    set_seed(args.seed)
    device = get_device()
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    num_classes = checkpoint["num_classes"]

    y = None
    if num_classes:
        y = torch.arange(num_classes, device=device).repeat_interleave(
            SAMPLES_PER_CLASS
        )
        model = ClassifierFreeGuidance(model, args.guidance_scale, num_classes)
    shape = (len(y) if y is not None else UNCONDITIONAL_SAMPLES, *IMAGE_SHAPE)

    if checkpoint["method"] == "ddpm":
        schedule = NoiseSchedule.create(checkpoint["schedule"]).to(device)
        images = ddim_sample(model, schedule, shape, args.steps, y=y, clip_x0=True)
    else:
        images = flow_sample(model, shape, args.steps, "heun", y=y, device=device)

    out = args.out or PROJECT_ROOT / "outputs" / f"{args.checkpoint.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    (ax,) = new_axes(1, size=6)
    show_images(images, ax, nrow=SAMPLES_PER_CLASS, title=args.checkpoint.stem)
    plt.savefig(out, dpi=100, bbox_inches="tight")
    logger.info(f"saved {len(images)} samples to {out}")


if __name__ == "__main__":
    main()
