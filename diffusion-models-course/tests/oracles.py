# tests/oracles.py
"""Exact models for a one-point data distribution: what a perfectly trained network outputs."""

import torch

from diffusion_course.schedules import NoiseSchedule


def single_point_oracle(schedule: NoiseSchedule, target: torch.Tensor):
    """The exact noise predictor when the data distribution is one point."""

    def model(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        steps = (t * schedule.num_steps).round().long()
        alpha_bar = schedule.alpha_bars[steps][:, None]
        return (x - alpha_bar.sqrt() * target) / (1 - alpha_bar).sqrt()

    return model
