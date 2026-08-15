"""1-D graph-structured decoder: decoding on the feature-space manifold.

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
not the full second-order wave recurrence from ESDM (which remains Phase 3).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.spectral_schedule import SpectralSchedule

__all__ = ["WaveReconstructionBlock", "GraphDecoder", "ClockGatedGraphDecoder"]


class WaveReconstructionBlock(nn.Module):
    """Wave-based reconstruction block on the ArrowSpace graph (1-D audio).

    Propagates information along the graph's smooth directions (U_q),
    weighted by the dispersion network (via c_spec which contains λ_chart).

    The block performs:
    1. Project feature activations onto the chart basis: Ĥ = H @ U_q
    2. Gate by dispersion-derived weights: g = σ(W @ c_spec)
    3. Lift back to feature space: H' = H + (Ĥ ⊙ g) @ U_q^T
    4. Apply a residual conv: out = ResBlock(H')

    This is the minimal graph-filter step. The full wave recurrence
    (Q_{t+1} = 2Q_t - Q_{t-1} - Δτ² L_F Q_t) remains Phase 3.

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

        self.feature_to_chart = nn.Conv1d(channels, feature_dim, 1)
        self.chart_to_feature = nn.Conv1d(feature_dim, channels, 1)
        self.chart_to_feature.weight.data.mul_(0.01)
        assert self.chart_to_feature.bias is not None
        self.chart_to_feature.bias.data.mul_(0.01)

        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, h: Tensor, c_spec: Tensor) -> Tensor:
        """Apply wave-based reconstruction.

        Parameters
        ----------
        h : Tensor (B, C, T)
            Feature activations (1-D temporal).
        c_spec : Tensor (B, 3*q)
            Spectral conditioning vector [ẽ, λ_chart, ν].

        Returns
        -------
        Tensor (B, C, T)
        """
        # Per-time-step projection to feature space: (B, C, T) -> (B, F, T)
        a = self.feature_to_chart(h)

        # Project to chart: (B, F, T) @ (F, q) -> (B, q, T)
        h_hat = torch.einsum("bft,fq->bqt", a, self.U_q)

        # Gate by dispersion: (B, 3q) -> (B, q)
        g = torch.sigmoid(self.gate(c_spec))

        # Gated projection (gate is clip-level, broadcast over time)
        gated = h_hat * g.unsqueeze(-1)

        # Lift back to feature space: (B, q, T) @ (q, F) -> (B, F, T)
        a_recon = torch.einsum("bqt,qf->bft", gated, self.U_q.T)

        # Map back to channel space: (B, F, T) -> (B, C, T)
        delta = self.chart_to_feature(a_recon)

        h_modulated = h + delta

        h_out = h + self.conv(self.act(self.norm(h_modulated)))

        return h_out


class GraphDecoder(nn.Module):
    """1-D topology-adaptive decoder using graph-structured reconstruction.

    Replaces the standard VAE decoder's unconstrained convolutions with
    WaveReconstructionBlock instances that decode along U_q directions,
    gated by the dispersion network.

    Parameters
    ----------
    latent_channels : int
        Spatial latent channels.
    out_channels : int
        Output audio channels (1 for mono).
    feature_dim : int
        ArrowSpace feature dimension F.
    base_channels : int
        Base width for conv layers.
    prior : ArrowSpacePrior
        The frozen prior providing U_q and λ_ED.
    upsample_strides : tuple[int, ...]
        Upsampling factors per stage. Default (2, 4, 5, 8) gives 320×
        total, matching EnCodec's 24 kHz stride.
    """

    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        feature_dim: int,
        base_channels: int,
        prior: ArrowSpacePrior,
        upsample_strides: tuple[int, ...] = (2, 4, 5, 8),
    ) -> None:
        super().__init__()
        self.prior = prior
        self.upsample_strides = upsample_strides

        ch = base_channels

        self.dec_in = nn.Conv1d(latent_channels, ch * 4, 1)

        # Build wave blocks and dec stages dynamically
        self.wave_blocks = nn.ModuleList()
        self.dec_stages = nn.ModuleList()

        # Channel progression: ch*4 -> ch*2 -> ch -> ch -> ch
        channel_steps = [ch * 4, ch * 2, ch, ch, ch]
        # Ensure enough steps for all stages
        while len(channel_steps) < len(upsample_strides) + 1:
            channel_steps.append(ch)

        for i, stride in enumerate(upsample_strides):
            in_ch = channel_steps[i]
            out_ch = channel_steps[i + 1]
            self.wave_blocks.append(
                WaveReconstructionBlock(
                    channels=in_ch, feature_dim=feature_dim, prior=prior
                )
            )
            self.dec_stages.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, 3, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                )
            )

        self.dec_out = nn.Conv1d(
            channel_steps[len(upsample_strides)], out_channels, 3, padding=1
        )

    def forward(self, z: Tensor, c_spec: Tensor) -> Tensor:
        """Decode 1-D latent under graph-structured reconstruction.

        Parameters
        ----------
        z : Tensor (B, latent_channels, T)
            1-D audio latent.
        c_spec : Tensor (B, 3*q)
            Spectral conditioning vector.

        Returns
        -------
        Tensor (B, out_channels, T * prod(upsample_strides))
        """
        h = self.dec_in(z)

        for i, stride in enumerate(self.upsample_strides):
            h = self.wave_blocks[i](h, c_spec)
            h = self.dec_stages[i](h)
            h = nn.functional.interpolate(h, scale_factor=stride, mode="nearest")

        x_hat = self.dec_out(h)
        return x_hat


class ClockGatedGraphDecoder(nn.Module):
    """1-D graph decoder with clock-gated decoding tempo.

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
    upsample_strides : tuple[int, ...]
        Upsampling factors per stage (default (2, 4, 5, 8) for EnCodec).
    """

    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        feature_dim: int,
        base_channels: int,
        prior: ArrowSpacePrior,
        spectral_schedule: SpectralSchedule,
        upsample_strides: tuple[int, ...] = (2, 4, 5, 8),
    ) -> None:
        super().__init__()
        self.prior = prior
        self.spectral_schedule = spectral_schedule
        self.upsample_strides = upsample_strides

        ch = base_channels

        self.dec_in = nn.Conv1d(latent_channels, ch * 4, 1)

        self.wave_blocks = nn.ModuleList()
        self.dec_stages = nn.ModuleList()

        channel_steps = [ch * 4, ch * 2, ch, ch, ch]
        while len(channel_steps) < len(upsample_strides) + 1:
            channel_steps.append(ch)

        for i, stride in enumerate(upsample_strides):
            in_ch = channel_steps[i]
            out_ch = channel_steps[i + 1]
            self.wave_blocks.append(
                WaveReconstructionBlock(
                    channels=in_ch, feature_dim=feature_dim, prior=prior
                )
            )
            self.dec_stages.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, 3, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                )
            )

        self.dec_out = nn.Conv1d(
            channel_steps[len(upsample_strides)], out_channels, 3, padding=1
        )

    def forward(
        self,
        z: Tensor,
        c_spec: Tensor,
        diffusion_time: Tensor | None = None,
    ) -> Tensor:
        """Decode with clock-gated tempo.

        Parameters
        ----------
        z : Tensor (B, latent_channels, T)
            1-D audio latent.
        c_spec : Tensor (B, 3*q)
            Spectral conditioning vector.
        diffusion_time : Tensor (scalar), optional
            Current diffusion time in [0, 1] (1 = high noise, 0 = clean).
            If None, defaults to 0.0 (full gate strength).

        Returns
        -------
        Tensor (B, out_channels, T * prod(upsample_strides))
        """
        if diffusion_time is None:
            diffusion_time = torch.tensor(0.0, device=z.device)

        ab_k = self.spectral_schedule.alpha_bar_k(diffusion_time)
        tempo = ab_k.mean()

        h = self.dec_in(z)

        for i, stride in enumerate(self.upsample_strides):
            h_raw = self.wave_blocks[i](h, c_spec)
            h = h + tempo * (h_raw - h)
            h = self.dec_stages[i](h)
            h = nn.functional.interpolate(h, scale_factor=stride, mode="nearest")

        x_hat = self.dec_out(h)
        return x_hat
