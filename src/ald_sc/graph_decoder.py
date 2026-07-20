"""Graph-structured decoder: decoding on the feature-space manifold.

This is the research contribution of ALD-SC. Standard VAEs decode via the
reparameterization trick — a continuous surface integral over a Gaussian
latent, computed via unconstrained convolutions. This module replaces that
with a decoder where:

- L_F (via U_q) defines the **reconstruction paths**: information propagates
  along the graph's smooth eigenvector directions, not through unconstrained
  convolutions.
- λ_ED defines the **energy allocation**: the dispersion network gates how
  much reconstruction energy each spectral mode receives.

The WaveReconstructionBlock is the graph-theoretic analogue of the VAE
reparameterization trick: it propagates information along the graph's smooth
directions, weighted by the dispersion network.

This is a minimal first implementation: a single graph-filter step per block,
not the full second-order wave recurrence from ESDM (which remains Phase 4).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.spectral_schedule import SpectralSchedule

__all__ = ["WaveReconstructionBlock", "GraphDecoder"]


class WaveReconstructionBlock(nn.Module):
    """Wave-based reconstruction block on the ArrowSpace graph.

    Propagates information along the graph's smooth directions (U_q),
    weighted by the dispersion network (via c_spec which contains λ_chart).

    The block performs:
    1. Project feature activations onto the chart basis: Ĥ = H @ U_q
    2. Gate by dispersion-derived weights: g = σ(W @ c_spec)
    3. Lift back to feature space: H' = H + (Ĥ ⊙ g) @ U_q^T
    4. Apply a residual conv: out = ResBlock(H')

    This is the minimal graph-filter step. The full wave recurrence
    (Q_{t+1} = 2Q_t - Q_{t-1} - Δτ² L_F Q_t) remains Phase 4.

    Parameters
    ----------
    channels : int
        Number of feature channels.
    feature_dim : int
        Dimension F of the ArrowSpace feature-space.
    prior : ArrowSpacePrior
        The frozen prior providing U_q.
    """

    def __init__(
        self,
        channels: int,
        feature_dim: int,
        prior: ArrowSpacePrior,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.feature_dim = feature_dim
        self.q = prior.q

        self.register_buffer("U_q", prior.U_q.clone().float())

        self.gate = nn.Linear(3 * prior.q, prior.q)
        nn.init.zeros_(self.gate.bias)

        self.feature_to_chart = nn.Linear(channels, feature_dim)
        self.chart_to_feature = nn.Linear(feature_dim, channels)

        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, h: Tensor, c_spec: Tensor) -> Tensor:
        """Apply wave-based reconstruction.

        Parameters
        ----------
        h : Tensor (B, C, H, W)
            Feature activations.
        c_spec : Tensor (B, 3*q)
            Spectral conditioning vector [ẽ, λ_chart, ν].

        Returns
        -------
        Tensor (B, C, H, W)
        """
        B, C, H, W = h.shape

        pooled = h.mean(dim=(2, 3))

        A = self.feature_to_chart(pooled)

        H_hat = A @ self.U_q

        g = torch.sigmoid(self.gate(c_spec))

        gated = H_hat * g

        A_recon = gated @ self.U_q.T

        delta = self.chart_to_feature(A_recon)

        delta = delta.unsqueeze(-1).unsqueeze(-1)

        h_modulated = h + delta

        h_out = h + self.conv(self.act(self.norm(h_modulated)))

        return h_out


class GraphDecoder(nn.Module):
    """Topology-adaptive decoder using graph-structured reconstruction.

    Replaces the standard VAE decoder's unconstrained convolutions with
    WaveReconstructionBlock instances that decode along U_q directions,
    gated by the dispersion network.

    Parameters
    ----------
    latent_channels : int
        Spatial latent channels.
    out_channels : int
        Output image channels.
    feature_dim : int
        ArrowSpace feature dimension F.
    base_channels : int
        Base width for conv layers.
    prior : ArrowSpacePrior
        The frozen prior providing U_q and λ_ED.
    """

    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        feature_dim: int,
        base_channels: int,
        prior: ArrowSpacePrior,
    ) -> None:
        super().__init__()
        self.prior = prior

        ch = base_channels

        self.dec_in = nn.Conv2d(latent_channels, ch * 4, 1)

        self.wave_block_1 = WaveReconstructionBlock(
            channels=ch * 4, feature_dim=feature_dim, prior=prior
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 2, 3, padding=1),
            nn.GroupNorm(8, ch * 2),
            nn.SiLU(),
        )

        self.wave_block_2 = WaveReconstructionBlock(
            channels=ch * 2, feature_dim=feature_dim, prior=prior
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
        )

        self.wave_block_3 = WaveReconstructionBlock(
            channels=ch, feature_dim=feature_dim, prior=prior
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
        )

        self.dec_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, z: Tensor, c_spec: Tensor) -> Tensor:
        """Decode spatial latent under graph-structured reconstruction.

        Parameters
        ----------
        z : Tensor (B, latent_channels, h, w)
            Spatial latent.
        c_spec : Tensor (B, 3*q)
            Spectral conditioning vector.

        Returns
        -------
        Tensor (B, out_channels, H, W)
        """
        h = self.dec_in(z)

        h = self.wave_block_1(h, c_spec)
        h = self.dec1(h)
        h = nn.functional.interpolate(h, scale_factor=2, mode="nearest")

        h = self.wave_block_2(h, c_spec)
        h = self.dec2(h)
        h = nn.functional.interpolate(h, scale_factor=2, mode="nearest")

        h = self.wave_block_3(h, c_spec)
        h = self.dec3(h)

        x_hat = self.dec_out(h)
        return x_hat


class ClockGatedGraphDecoder(nn.Module):
    """Graph decoder with clock-gated decoding tempo.

    The SpectralSchedule governs when each spectral mode resolves during
    decoding. The clock modulates the strength of each wave block's gate
    based on the diffusion time:

    - Early in the denoising process (t near 1.0, high noise): modes are
      unresolved, gates are weak (the decoder does minimal graph-structured
      reconstruction).
    - Late in the denoising process (t near 0.0, low noise): modes are
      active, gates are strong (the decoder fully uses the graph structure).

    This implements the Barontini principle within the decoder: the graph
    geometry governs *when* reconstruction effort is allocated, not just
    *where*.

    Parameters
    ----------
    latent_channels : int
    out_channels : int
    feature_dim : int
        ArrowSpace feature dimension F.
    base_channels : int
    prior : ArrowSpacePrior
        Frozen prior providing U_q and λ_ED.
    spectral_schedule : SpectralSchedule
        Frozen schedule providing per-mode ᾱ_k(t) for tempo modulation.
    """

    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        feature_dim: int,
        base_channels: int,
        prior: ArrowSpacePrior,
        spectral_schedule: SpectralSchedule,
    ) -> None:
        super().__init__()
        self.prior = prior
        self.spectral_schedule = spectral_schedule

        ch = base_channels

        self.dec_in = nn.Conv2d(latent_channels, ch * 4, 1)

        self.wave_block_1 = WaveReconstructionBlock(
            channels=ch * 4, feature_dim=feature_dim, prior=prior
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 2, 3, padding=1),
            nn.GroupNorm(8, ch * 2),
            nn.SiLU(),
        )

        self.wave_block_2 = WaveReconstructionBlock(
            channels=ch * 2, feature_dim=feature_dim, prior=prior
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
        )

        self.wave_block_3 = WaveReconstructionBlock(
            channels=ch, feature_dim=feature_dim, prior=prior
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
        )

        self.dec_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        z: Tensor,
        c_spec: Tensor,
        diffusion_time: Tensor | None = None,
    ) -> Tensor:
        """Decode with clock-gated tempo.

        Parameters
        ----------
        z : Tensor (B, latent_channels, h, w)
            Spatial latent.
        c_spec : Tensor (B, 3*q)
            Spectral conditioning vector.
        diffusion_time : Tensor (scalar), optional
            Current diffusion time in [0, 1] (1 = high noise, 0 = clean).
            If None, defaults to 0.0 (full gate strength).

        Returns
        -------
        Tensor (B, out_channels, H, W)
        """
        if diffusion_time is None:
            diffusion_time = torch.tensor(0.0, device=z.device)

        ab_k = self.spectral_schedule.alpha_bar_k(diffusion_time)
        tempo = ab_k.mean()

        h = self.dec_in(z)

        h_raw = self.wave_block_1(h, c_spec)
        h = h + tempo * (h_raw - h)
        h = self.dec1(h)
        h = nn.functional.interpolate(h, scale_factor=2, mode="nearest")

        h_raw = self.wave_block_2(h, c_spec)
        h = h + tempo * (h_raw - h)
        h = self.dec2(h)
        h = nn.functional.interpolate(h, scale_factor=2, mode="nearest")

        h_raw = self.wave_block_3(h, c_spec)
        h = h + tempo * (h_raw - h)
        h = self.dec3(h)

        x_hat = self.dec_out(h)
        return x_hat
