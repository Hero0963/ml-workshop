# src/diffusion_course/data.py
"""Datasets: 2D toy distributions (fast on CPU) and MNIST-style images.

All data is scaled to roughly unit size, because the diffusion forward process mixes the
data with unit-variance Gaussian noise -- data far larger or smaller than that would make
the noise schedule mean something different.
"""

import math
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

TOY_DATASETS = ("swiss_roll", "moons", "checkerboard", "gaussian_mixture")
TOY_NOISE_STD = 0.05
SWISS_ROLL_SCALE = 0.2
MOONS_SCALE = 1.5
IMAGE_DATASETS = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST}
NUM_IMAGE_CLASSES = 10


class GaussianMixture:
    """An isotropic Gaussian mixture whose density and score are known in closed form.

    It is the one data distribution where we can check a learned score against the truth:
    adding Gaussian noise to a Gaussian mixture gives another Gaussian mixture.
    """

    def __init__(
        self, means: torch.Tensor, std: float, weights: torch.Tensor | None = None
    ) -> None:
        self.means = means.float()
        self.std = float(std)
        num_components = means.shape[0]
        self.weights = (
            torch.full((num_components,), 1.0 / num_components)
            if weights is None
            else weights.float() / weights.sum()
        )

    @classmethod
    def ring(
        cls, num_components: int = 8, radius: float = 2.0, std: float = 0.15
    ) -> "GaussianMixture":
        angles = torch.arange(num_components) * 2 * math.pi / num_components
        means = radius * torch.stack([angles.cos(), angles.sin()], dim=1)
        return cls(means, std)

    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        idx = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        noise = torch.randn(n, self.means.shape[1], generator=generator)
        return self.means[idx] + self.std * noise

    def _component_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.shape[1]
        sq_dist = torch.cdist(x, self.means.to(x)) ** 2
        log_norm = -0.5 * dim * math.log(2 * math.pi * self.std**2)
        return self.weights.to(x).log() + log_norm - sq_dist / (2 * self.std**2)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(self._component_log_probs(x), dim=1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """grad_x log p(x): a responsibility-weighted pull towards each component mean."""
        resp = torch.softmax(self._component_log_probs(x), dim=1)
        pull = (self.means.to(x)[None, :, :] - x[:, None, :]) / self.std**2
        return (resp[:, :, None] * pull).sum(dim=1)

    def diffused(self, alpha_bar: float) -> "GaussianMixture":
        """The distribution of x_t = sqrt(alpha_bar) x_0 + sqrt(1 - alpha_bar) eps."""
        std = math.sqrt(alpha_bar * self.std**2 + (1 - alpha_bar))
        return GaussianMixture(self.means * math.sqrt(alpha_bar), std, self.weights)


def make_toy_data(
    name: str, n: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Sample n points of shape (n, 2) from one of TOY_DATASETS."""
    if name == "gaussian_mixture":
        return GaussianMixture.ring().sample(n, generator)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=generator)

    if name == "swiss_roll":
        t = 1.5 * math.pi * (1 + 2 * rand(n))
        points = SWISS_ROLL_SCALE * torch.stack([t * t.cos(), t * t.sin()], dim=1)
    elif name == "moons":
        theta = math.pi * rand(n)
        upper = rand(n) < 0.5
        x = torch.where(upper, theta.cos(), 1 - theta.cos())
        y = torch.where(upper, theta.sin(), 0.5 - theta.sin())
        points = MOONS_SCALE * (torch.stack([x, y], dim=1) - torch.tensor([0.5, 0.25]))
    elif name == "checkerboard":
        x = 4 * rand(n) - 2
        y = rand(n) - 2 * torch.randint(0, 2, (n,), generator=generator).float()
        y = y + torch.floor(x) % 2
        points = torch.stack([x, y], dim=1)
    else:
        raise ValueError(f"unknown toy dataset {name!r}; choose from {TOY_DATASETS}")
    return points + TOY_NOISE_STD * torch.randn(n, 2, generator=generator)


def toy_batches(
    name: str, batch_size: int, seed: int = 0
) -> Iterator[tuple[torch.Tensor, None]]:
    """An endless stream of fresh toy batches (the toy datasets are infinite)."""
    generator = torch.Generator().manual_seed(seed)
    while True:
        yield make_toy_data(name, batch_size, generator), None


def image_dataset(
    name: str = "mnist", train: bool = True, root: Path = DATA_DIR
) -> datasets.VisionDataset:
    """MNIST or Fashion-MNIST as 1x28x28 tensors scaled to [-1, 1]; downloads on first use."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    return IMAGE_DATASETS[name](root, train=train, download=True, transform=transform)


def image_loader(
    name: str = "mnist",
    batch_size: int = 128,
    train: bool = True,
    root: Path = DATA_DIR,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        image_dataset(name, train, root),
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        num_workers=num_workers,
    )


def infinite_batches(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Cycle through a DataLoader forever, reshuffling at every pass."""
    while True:
        yield from loader


def to_unit_range(images: torch.Tensor) -> torch.Tensor:
    """Map model-space images in [-1, 1] back to [0, 1] for display."""
    return ((images + 1) / 2).clamp(0, 1)
