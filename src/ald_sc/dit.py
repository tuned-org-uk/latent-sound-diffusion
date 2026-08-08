"""Minimal 1-D DiT (Diffusion Transformer) denoiser for audio latents.

A single-layer transformer for the boilerplate.  Swap for a larger
architecture when scaling up.

Operates on 1-D audio latents z ∈ (B, C, T) produced by a frozen EnCodec
encoder.  Patchify uses Conv1d over the temporal axis; the AdaLN/CFG
conditioning scaffold is unchanged from the 2-D image version.
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
    """A minimal 1-D DiT denoiser for audio latent diffusion.

    Treats the 1-D audio latent as a sequence of temporal patches.
    Conditioned on timestep, optional text embeddings, and optional
    spectral-chart tokens via AdaLN.

    For Phase 1 sound generation, the model operates unconditionally
    (time-only AdaLN).  The ``c_spec`` and ``text_emb`` hooks are
    retained but default to ``None`` (unconditional).

    Parameters
    ----------
    latent_channels : int
        Channels of the 1-D latent z (e.g. 128 for EnCodec).
    latent_length : int
        Temporal length of the latent (e.g. 375 for 5s @ 24kHz).
    patch_size : int
        Patch size for temporal tokenisation.
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
    cfg_dropout : float
        Probability of dropping c_spec during training (classifier-free
        guidance).
    """

    def __init__(
        self,
        latent_channels: int = 128,
        latent_length: int = 375,
        patch_size: int = 8,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        text_dim: int = 0,
        spec_dim: int = 768,  # 3 * 256
        cfg_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cfg_dropout = cfg_dropout
        self.latent_channels = latent_channels
        self.latent_length = latent_length
        self.patch_size = patch_size
        self.dim = dim

        # latent_shape attribute for samplers: (channels, length)
        self.latent_shape = (latent_channels, latent_length)

        # 1-D Patch embedding
        self.patch_embed = nn.Conv1d(
            latent_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        # Use ceil division so pos_embed covers padded length
        num_patches = math.ceil(latent_length / patch_size)
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
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, cond_dim=dim) for _ in range(depth)])

        # Output
        self.final_norm = nn.LayerNorm(dim)
        self.final_proj = nn.Linear(dim, latent_channels * patch_size)

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
        z_t : Tensor (B, C, T)
            Noised 1-D audio latent.
        t : Tensor (B,)
            Timestep indices.
        text_emb : Tensor (B, text_dim), optional
            Text embedding vector.
        c_spec : Tensor (B, spec_dim), optional
            Spectral conditioning vector.

        Returns
        -------
        Tensor (B, C, T)
            Predicted velocity v.
        """
        B, _, T = z_t.shape
        ps = self.patch_size

        # Pad temporal axis to be divisible by patch_size
        pad_len = (ps - T % ps) % ps
        if pad_len > 0:
            z_t = nn.functional.pad(z_t, (0, pad_len))

        # 1-D Patchify: (B, C, T_pad) -> (B, dim, N) -> (B, N, dim)
        h = self.patch_embed(z_t)  # (B, dim, N)
        h = h.transpose(1, 2)  # (B, N, dim)
        h = h + self.pos_embed

        # Classifier-free guidance dropout
        if self.training and c_spec is not None and self.cfg_dropout > 0:
            mask = torch.rand(B, device=z_t.device) < self.cfg_dropout
            c_spec = c_spec.clone()
            c_spec[mask] = 0.0

        # Conditioning
        cond = self.time_emb(t)  # (B, dim)
        if text_emb is not None and hasattr(self, "text_proj"):
            cond = torch.cat([cond, self.text_proj(text_emb)], dim=-1)
        if c_spec is not None:
            cond = torch.cat([cond, self.spec_proj(c_spec)], dim=-1)
        else:
            # Unconditional: zero spectral conditioning
            zero_spec = torch.zeros(B, self.spec_proj.in_features, device=z_t.device)
            cond = torch.cat([cond, self.spec_proj(zero_spec)], dim=-1)
        cond = self.cond_fuse(cond)  # (B, dim)

        for block in self.blocks:
            h = block(h, cond)

        h = self.final_norm(h)
        h = self.final_proj(h)  # (B, N, C*patch)

        # 1-D Unpatchify: (B, N, C*patch) -> (B, C, T_pad)
        h = h.transpose(1, 2)  # (B, C*patch, N)
        h = h.reshape(B, self.latent_channels, ps, -1)  # (B, C, patch, N)
        h = h.reshape(B, self.latent_channels, -1)  # (B, C, T_pad)

        # Crop to original length
        if pad_len > 0:
            h = h[:, :, :T]

        return h
