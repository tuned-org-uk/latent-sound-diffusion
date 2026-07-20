"""Spectral VAE with dual-head encoder and topology-adaptive decoder.

The encoder produces two coupled outputs:
- z: spatial latent (standard VAE latent for diffusion)
- A: feature field for spectral chart extraction

The decoder reconstructs pixels under spectral conditioning from c_spec,
using spectrally gated up-blocks that project onto the ArrowSpace chart basis.

This module must not add diffusion logic (per AGENTS.md §11).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ald_sc.arrow_prior import ArrowSpacePrior

__all__ = ["SpectralVAE"]


class ResBlock(nn.Module):
    """Simple residual block with GroupNorm and SiLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = nn.functional.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.skip(x) + h


class SpectralVAE(nn.Module):
    """Spectral VAE with dual-head encoder and topology-adaptive decoder.

    Parameters
    ----------
    in_channels : int
        Input image channels (e.g. 3 for RGB).
    latent_channels : int
        Spatial latent channels.
    feature_dim : int
        Feature field dimension F (must match the prior).
    base_channels : int
        Base width for conv layers.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        feature_dim: int = 256,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.feature_dim = feature_dim
        self.base_channels = base_channels

        ch = base_channels

        # Shared encoder trunk
        self.enc1 = ResBlock(in_channels, ch)
        self.enc2 = ResBlock(ch, ch * 2)
        self.enc3 = ResBlock(ch * 2, ch * 4)

        # Spatial head: z = (mu, logvar)
        self.spatial_head = nn.Conv2d(ch * 4, latent_channels * 2, 1)

        # Feature head: A (global pooled feature field)
        self.feature_head = nn.AdaptiveAvgPool2d(1)
        self.feature_proj = nn.Linear(ch * 4, feature_dim)

        # Decoder
        self.dec_in = nn.Conv2d(latent_channels, ch * 4, 1)
        self.dec1 = ResBlock(ch * 4, ch * 4)
        self.dec2 = ResBlock(ch * 4, ch * 2)
        self.dec3 = ResBlock(ch * 2, ch)
        self.dec_out = nn.Conv2d(ch, in_channels, 3, padding=1)

        # Spectral gating: g = sigma(W * c_spec + b)
        # c_spec dim = 3 * q, but we don't know q at init time,
        # so we build the gate lazily on first forward.
        self._spectral_gate: nn.Linear | None = None

        self._last_mu: Tensor | None = None
        self._last_logvar: Tensor | None = None

    def _get_spectral_gate(self, c_spec_dim: int, device: torch.device) -> nn.Linear:
        if self._spectral_gate is None:
            self._spectral_gate = nn.Linear(c_spec_dim, self.base_channels * 4).to(
                device
            )
            nn.init.zeros_(self._spectral_gate.bias)
        return self._spectral_gate

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode image to (z, A, mu, logvar).

        Returns
        -------
        z : Tensor (B, latent_channels, h, w)
            Sampled spatial latent.
        A : Tensor (B, F)
            Feature field (global pooled).
        mu : Tensor (B, latent_channels, h, w)
        logvar : Tensor (B, latent_channels, h, w)
        """
        h = self.enc1(x)
        h = nn.functional.avg_pool2d(h, 2)
        h = self.enc2(h)
        h = nn.functional.avg_pool2d(h, 2)
        h = self.enc3(h)

        # Spatial head
        z_params = self.spatial_head(h)
        mu, logvar = z_params.chunk(2, dim=1)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)

        # Feature head
        A_pooled = self.feature_head(h).squeeze(-1).squeeze(-1)
        A = self.feature_proj(A_pooled)

        self._last_mu = mu
        self._last_logvar = logvar
        return z, A, mu, logvar

    def decode(self, z: Tensor, c_spec: Tensor, prior: ArrowSpacePrior) -> Tensor:
        """Decode spatial latent under spectral conditioning.

        Parameters
        ----------
        z : Tensor (B, latent_channels, h, w)
            Spatial latent.
        c_spec : Tensor (B, 3*q)
            Spectral conditioning vector.
        prior : ArrowSpacePrior
            Frozen prior for spectral gating.

        Returns
        -------
        Tensor (B, in_channels, H, W)
            Reconstructed image.
        """
        h = self.dec_in(z)

        # Spectral gating at the top decoder level
        gate_proj = self._get_spectral_gate(c_spec.shape[-1], h.device)
        g = torch.sigmoid(gate_proj(c_spec))
        g = g.unsqueeze(-1).unsqueeze(-1)
        h = h * (1 + g)

        h = self.dec1(h)
        h = nn.functional.interpolate(h, scale_factor=2, mode="nearest")
        h = self.dec2(h)
        h = nn.functional.interpolate(h, scale_factor=2, mode="nearest")
        h = self.dec3(h)
        x_hat = self.dec_out(h)
        return x_hat

    def forward(
        self, x: Tensor, prior: ArrowSpacePrior
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full encode-decode pass.

        Returns
        -------
        z : Tensor (B, latent_channels, h, w)
        A : Tensor (B, F)
        c_spec : Tensor (B, 3*q)
        x_hat : Tensor (B, in_channels, H, W)
        """
        z, A, mu, logvar = self.encode(x)
        c_spec = prior.chart_energy_descriptor(A)
        x_hat = self.decode(z, c_spec, prior)
        return z, A, c_spec, x_hat

    def kl_loss(self) -> Tensor:
        """Standard VAE KL from the last forward pass."""
        assert self._last_mu is not None and self._last_logvar is not None
        mu = self._last_mu
        logvar = self._last_logvar
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
