# src/diffusion_course/models/dit.py
"""A tiny Diffusion Transformer (Peebles & Xie, 2022) for 28x28 images.

The image is cut into patches, each patch becomes a token, and a plain Transformer
processes the tokens. Time and class enter through adaLN-Zero: every block's LayerNorm
gets a shift, a scale and a residual gate computed from the conditioning vector, and
the gates start at zero so each block starts out as the identity.
"""

import torch
import torch.nn.functional as F
from torch import nn

from diffusion_course.models.embeddings import LabelEmbedding, TimeEmbedding

POS_EMBED_INIT_STD = 0.02


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim)
        )
        self.ada_ln = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada_ln.weight)
        nn.init.zeros_(self.ada_ln.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada_ln(F.silu(cond)).chunk(
            6, dim=1
        )
        h = modulate(self.norm1(x), shift1, scale1)
        x = x + gate1[:, None, :] * self.attn(h, h, h, need_weights=False)[0]
        h = modulate(self.norm2(x), shift2, scale2)
        return x + gate2[:, None, :] * self.mlp(h)


class TinyDiT(nn.Module):
    """model(x, t, y) for images of shape (B, C, H, W) with H, W divisible by patch_size."""

    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid = image_size // patch_size
        self.patchify = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(
            POS_EMBED_INIT_STD * torch.randn(1, self.grid * self.grid, dim)
        )
        self.time_embed = TimeEmbedding(dim)
        self.label_embed = (
            LabelEmbedding(num_classes, dim) if num_classes is not None else None
        )
        self.blocks = nn.ModuleList(
            DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada_ln = nn.Linear(dim, 2 * dim)
        self.final_proj = nn.Linear(dim, patch_size * patch_size * in_channels)
        for layer in (self.final_ada_ln, self.final_proj):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, p*p*C) tokens back to a (B, C, H, W) image."""
        b, p, c, g = tokens.shape[0], self.patch_size, self.in_channels, self.grid
        x = tokens.reshape(b, g, g, p, p, c).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(b, c, g * p, g * p)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        cond = self.time_embed(t)
        if self.label_embed is not None:
            cond = cond + self.label_embed(y, x.shape[0])
        tokens = self.patchify(x).flatten(2).transpose(1, 2) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens, cond)
        shift, scale = self.final_ada_ln(F.silu(cond)).chunk(2, dim=1)
        tokens = self.final_proj(modulate(self.final_norm(tokens), shift, scale))
        return self.unpatchify(tokens)
