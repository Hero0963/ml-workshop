# src/diffusion_course/evaluation.py
"""A small digit classifier used as a cheap judge of generated MNIST samples.

Real papers report FID, which needs an ImageNet-trained Inception network and tens of
thousands of samples. For 28x28 digits a classifier trained in a minute answers the two
questions the labs care about: does a sample look like *some* digit (confidence), and does
it look like the digit we *asked for* (accuracy)?
"""

import math
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from diffusion_course.data import NUM_IMAGE_CLASSES, image_loader, infinite_batches
from diffusion_course.training import TrainResult, train
from diffusion_course.utils import CHECKPOINT_DIR

CLASSIFIER_LR = 1e-3


class DigitClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_IMAGE_CLASSES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_classifier(
    batches: Iterator[tuple[torch.Tensor, torch.Tensor]],
    num_steps: int = 600,
    device: torch.device | str = "cpu",
) -> tuple[DigitClassifier, TrainResult]:
    """Train on (image, label) batches; returns the classifier in eval mode."""
    classifier = DigitClassifier()

    def loss_fn(
        model: nn.Module, x: torch.Tensor, y: torch.Tensor | None
    ) -> torch.Tensor:
        return F.cross_entropy(model(x), y)

    result = train(
        classifier,
        loss_fn,
        batches,
        num_steps,
        lr=CLASSIFIER_LR,
        ema_decay=None,
        device=device,
        log_every=0,
    )
    return classifier, result


@torch.no_grad()
def judge_samples(
    classifier: nn.Module,
    images: torch.Tensor,
    requested: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarise a batch of generated digits.

    confidence: mean probability of the predicted class (does it look like a digit?)
    class_entropy: entropy of the predicted-class histogram divided by log(10); 1.0 means
        all ten digits appear equally often, 0.0 means every sample is the same digit.
    accuracy: fraction predicted as the requested class (only with ``requested``).
    """
    device = next(classifier.parameters()).device
    probs = torch.softmax(classifier(images.to(device)), dim=1)
    confidence, predicted = probs.max(dim=1)
    histogram = torch.bincount(predicted, minlength=probs.shape[1]).float()
    histogram = histogram / histogram.sum()
    entropy = -(histogram[histogram > 0] * histogram[histogram > 0].log()).sum()
    report = {
        "confidence": confidence.mean().item(),
        "class_entropy": entropy.item() / math.log(probs.shape[1]),
    }
    if requested is not None:
        report["accuracy"] = (predicted == requested.to(device)).float().mean().item()
    return report


def get_digit_classifier(
    device: torch.device | str = "cpu", num_steps: int = 600
) -> DigitClassifier:
    """Load the cached MNIST judge, or train it once (about a minute on a CPU) and cache it."""
    path = CHECKPOINT_DIR / "digit_classifier.pt"
    classifier = DigitClassifier()
    if path.exists():
        classifier.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        return classifier.to(device).eval()
    classifier, _ = train_classifier(
        infinite_batches(image_loader("mnist", 128)), num_steps, device
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), path)
    return classifier.eval()
