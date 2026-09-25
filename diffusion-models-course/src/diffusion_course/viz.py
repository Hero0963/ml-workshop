# src/diffusion_course/viz.py
"""Plotting helpers so the lab notebooks can stay focused on the ideas."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from torchvision.utils import make_grid

from diffusion_course.data import to_unit_range

TOY_LIMIT = 3.5


def new_axes(ncols: int = 1, size: float = 3.2, nrows: int = 1) -> list[Axes]:
    _, axes = plt.subplots(
        nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False
    )
    return list(axes.flat)


def plot_points(
    points: torch.Tensor,
    ax: Axes,
    title: str = "",
    limit: float = TOY_LIMIT,
    size: float = 1.0,
    color: str = "tab:blue",
) -> None:
    points = points.detach().cpu()
    ax.scatter(points[:, 0], points[:, 1], s=size, alpha=0.5, color=color)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def plot_trajectories(
    trajectory: torch.Tensor, ax: Axes, num_paths: int = 40, limit: float = TOY_LIMIT
) -> None:
    """trajectory: (num_steps, N, 2). Grey paths, black start (noise), red end (sample)."""
    paths = trajectory[:, :num_paths].detach().cpu()
    for i in range(paths.shape[1]):
        ax.plot(paths[:, i, 0], paths[:, i, 1], color="grey", lw=0.6, alpha=0.7)
    ax.scatter(paths[0, :, 0], paths[0, :, 1], s=6, color="black", label="start")
    ax.scatter(paths[-1, :, 0], paths[-1, :, 1], s=6, color="tab:red", label="end")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")


def plot_vector_field(
    field_fn: Callable[[torch.Tensor], torch.Tensor],
    ax: Axes,
    limit: float = TOY_LIMIT,
    resolution: int = 21,
    title: str = "",
) -> None:
    """Draw arrows of field_fn (N, 2) -> (N, 2) on a regular grid."""
    axis = torch.linspace(-limit, limit, resolution)
    xs, ys = torch.meshgrid(axis, axis, indexing="xy")
    grid = torch.stack([xs.flatten(), ys.flatten()], dim=1)
    with torch.no_grad():
        arrows = field_fn(grid).cpu()
    ax.quiver(
        grid[:, 0],
        grid[:, 1],
        arrows[:, 0],
        arrows[:, 1],
        color="tab:purple",
        alpha=0.8,
    )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def show_images(images: torch.Tensor, ax: Axes, nrow: int = 8, title: str = "") -> None:
    """images: (N, C, H, W) in model space [-1, 1]."""
    grid = make_grid(to_unit_range(images.detach().cpu()), nrow=nrow, padding=1)
    ax.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def plot_loss(
    losses: list[float], ax: Axes, window: int = 100, title: str = "training loss"
) -> None:
    values = torch.tensor(losses)
    if len(values) >= window:
        values = values.unfold(0, window, 1).mean(dim=1)
    ax.plot(values)
    ax.set_xlabel("step")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=9)
