"""Minimal DiT (Diffusion Transformer) denoiser conditioned on text and
spectral-chart tokens.

A single-layer transformer for the boilerplate.  Swap for a larger architecture
when scaling up.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["SinusoidalTimeEmb", "AdaLN", "DiTBlock", "MinimalDiT"]


class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal timestep embedding followed by a small MLP."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb)


class AdaLN(nn.Module):
    """Adaptive Layer Norm: produces scale and shift from a conditioning vector."""

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift, scale = self.proj(cond).chunk(2, dim=-1)
        scale = scale[:, None, :]
        shift = shift[:, None, :]
        return self.norm(x) * (1 + scale) + shift


class DiTBlock(nn.Module):
    """Single DiT transformer block with adaptive layer norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        cond_dim = cond_dim or dim
        self.adaln = AdaLN(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.adaln(x, cond)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class MinimalDiT(nn.Module):
    """A minimal DiT denoiser for latent diffusion.

    Treats the spatial latent as a sequence of patches.  Conditioned on
    timestep, text embeddings, and spectral-chart tokens via AdaLN.

    Parameters
    ----------
    latent_channels : int
        Channels of the spatial latent z.
    latent_size : int
        Spatial size of the latent (assumed square, e.g. 32 or 64).
    patch_size : int
        Patch size for latent tokenisation.
    dim : int
        Transformer hidden dimension.
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads in each block.
    text_dim : int
        Dimension of text embedding input (0 to disable).
    spec_dim : int
        Dimension of spectral conditioning vector c_spec.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 32,
        patch_size: int = 2,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        text_dim: int = 0,
        spec_dim: int = 768,  # 3 * 256
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.dim = dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            latent_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (latent_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Time embedding
        self.time_emb = SinusoidalTimeEmb(dim)

        # Conditioning fusion: time + text + spectral
        cond_parts = [dim]  # time
        if text_dim > 0:
            self.text_proj = nn.Linear(text_dim, dim)
            cond_parts.append(dim)
        self.spec_proj = nn.Linear(spec_dim, dim)
        cond_parts.append(dim)
        self.cond_fuse = nn.Linear(sum(cond_parts), dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, num_heads, cond_dim=dim) for _ in range(depth)]
        )

        # Output
        self.final_norm = nn.LayerNorm(dim)
        self.final_proj = nn.Linear(dim, latent_channels * patch_size * patch_size)

    def forward(
        self,
        z_t: Tensor,
        t: Tensor,
        text_emb: Tensor | None = None,
        c_spec: Tensor | None = None,
    ) -> Tensor:
        """Predict v (velocity) or noise.

        Parameters
        ----------
        z_t : Tensor (B, c, h, w)
            Noised latent.
        t : Tensor (B,)
            Timestep indices.
        text_emb : Tensor (B, text_dim), optional
            Text embedding vector.
        c_spec : Tensor (B, spec_dim), optional
            Spectral conditioning vector.

        Returns
        -------
        Tensor (B, c, h, w)
            Predicted velocity v.
        """
        B = z_t.shape[0]

        # Patchify
        h = self.patch_embed(z_t)  # (B, dim, h/ps, w/ps)
        h = h.flatten(2).transpose(1, 2)  # (B, N, dim)
        h = h + self.pos_embed

        # Conditioning
        cond = self.time_emb(t)  # (B, dim)
        if text_emb is not None and hasattr(self, "text_proj"):
            cond = torch.cat([cond, self.text_proj(text_emb)], dim=-1)
        if c_spec is not None:
            cond = torch.cat([cond, self.spec_proj(c_spec)], dim=-1)
        cond = self.cond_fuse(cond)  # (B, dim)

        for block in self.blocks:
            h = block(h, cond)

        h = self.final_norm(h)
        h = self.final_proj(h)  # (B, N, c*ps*ps)

        # Unpatchify
        ps = self.patch_size
        h = h.transpose(1, 2)  # (B, c*ps*ps, N)
        h = h.reshape(
            B,
            self.latent_channels,
            ps,
            ps,
            self.latent_size // ps,
            self.latent_size // ps,
        )
        h = h.permute(0, 1, 4, 2, 5, 3)  # (B, c, h/ps, ps, w/ps, ps)
        h = h.reshape(B, self.latent_channels, self.latent_size, self.latent_size)
        return h
