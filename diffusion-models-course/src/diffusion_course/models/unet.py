# src/diffusion_course/models/unet.py
"""A compact DDPM-style U-Net for 28x28 images (MNIST, Fashion-MNIST).

Same ingredients as the U-Net of Ho et al. (2020), scaled down: residual blocks with
GroupNorm, the time (and class) embedding added inside every block, self-attention at
the lowest resolution, and skip connections from each encoder level to its decoder twin.

    28x28 --down--> 14x14 --down--> 7x7 (attention) --> middle --> up, up (+ skips)
"""

import torch
import torch.nn.functional as F
from torch import nn

from diffusion_course.models.embeddings import LabelEmbedding, TimeEmbedding

NUM_GROUPS = 8


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(NUM_GROUPS, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.cond = nn.Linear(cond_dim, out_ch)
        self.norm2 = nn.GroupNorm(NUM_GROUPS, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond(cond)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Every pixel attends to every other pixel -- affordable only at 7x7."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(NUM_GROUPS, channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = (
            self.qkv(self.norm(x))
            .reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
            .unbind(1)
        )
        out = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        )
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class Level(nn.Module):
    """Two residual blocks, optionally with attention between them."""

    def __init__(
        self, in_ch: int, out_ch: int, cond_dim: int, attention: bool, dropout: float
    ) -> None:
        super().__init__()
        self.block1 = ResBlock(in_ch, out_ch, cond_dim, dropout)
        self.attn = SelfAttention2d(out_ch) if attention else nn.Identity()
        self.block2 = ResBlock(out_ch, out_ch, cond_dim, dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.block2(self.attn(self.block1(x, cond)), cond)


class UNet(nn.Module):
    """model(x, t, y) for images of shape (B, C, H, W); H and W must be divisible by 4."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 2),
        attention_levels: tuple[int, ...] = (2,),
        num_classes: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = [base_channels * m for m in channel_mults]
        cond_dim = 4 * base_channels
        self.time_embed = TimeEmbedding(cond_dim)
        self.label_embed = (
            LabelEmbedding(num_classes, cond_dim) if num_classes is not None else None
        )
        self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        prev = channels[0]
        last = len(channels) - 1
        for i, ch in enumerate(channels):
            self.encoder.append(
                Level(prev, ch, cond_dim, i in attention_levels, dropout)
            )
            self.downsample.append(
                nn.Conv2d(ch, ch, 3, stride=2, padding=1) if i < last else nn.Identity()
            )
            prev = ch

        self.middle = Level(prev, prev, cond_dim, attention=True, dropout=dropout)

        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in reversed(range(len(channels))):
            ch = channels[i]
            self.decoder.append(
                Level(prev + ch, ch, cond_dim, i in attention_levels, dropout)
            )
            self.upsample.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2), nn.Conv2d(ch, ch, 3, padding=1)
                )
                if i > 0
                else nn.Identity()
            )
            prev = ch

        self.head = nn.Sequential(
            nn.GroupNorm(NUM_GROUPS, prev),
            nn.SiLU(),
            nn.Conv2d(prev, in_channels, 3, padding=1),
        )
        # Start as a network that predicts zero: training begins from a sensible baseline.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        cond = self.time_embed(t)
        if self.label_embed is not None:
            cond = cond + self.label_embed(y, x.shape[0])

        h = self.stem(x)
        skips = []
        for level, down in zip(self.encoder, self.downsample):
            h = level(h, cond)
            skips.append(h)
            h = down(h)

        h = self.middle(h, cond)

        for level, up in zip(self.decoder, self.upsample):
            h = level(torch.cat([h, skips.pop()], dim=1), cond)
            h = up(h)
        return self.head(h)
