"""ALD-SC audio loss components.

Computes the spectral loss terms for audio decoder training:
- L_rec: reconstruction (L1 + multi-scale STFT)
- L_chart: spectral chart consistency
- L_smooth: off-manifold penalty

The KL loss is retained for interface compatibility but is unused for
audio (EnCodec encoder is deterministic, not a VAE).

This module must not import trainer or model modules except for type hints
(per AGENTS.md §11).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ald_sc.arrow_prior import ArrowSpacePrior

__all__ = ["ALDSCLoss"]


class ALDSCLoss:
    """Composite loss for ALD-SC audio decoder training.

    Parameters
    ----------
    prior : ArrowSpacePrior
        The frozen ArrowSpace prior for spectral computations.
    lambda_rec : float
        Weight for reconstruction loss (L1 component).
    lambda_stft : float
        Weight for multi-scale STFT spectral loss.
    lambda_chart : float
        Weight for spectral chart consistency loss.
    lambda_smooth : float
        Weight for off-manifold penalty.
    lambda_kl : float
        Weight for VAE KL regularisation (unused for audio, kept for compat).
    stft_fft_sizes : tuple[int, ...]
        FFT sizes for multi-scale STFT loss.
    """

    def __init__(
        self,
        prior: ArrowSpacePrior,
        lambda_rec: float = 1.0,
        lambda_stft: float = 1.0,
        lambda_chart: float = 0.5,
        lambda_smooth: float = 0.1,
        lambda_kl: float = 0.0,
        stft_fft_sizes: tuple[int, ...] = (512, 1024, 2048),
    ) -> None:
        self.prior = prior
        self.lambda_rec = lambda_rec
        self.lambda_stft = lambda_stft
        self.lambda_chart = lambda_chart
        self.lambda_smooth = lambda_smooth
        self.lambda_kl = lambda_kl
        self.stft_fft_sizes = stft_fft_sizes
        self._windows: dict[int, Tensor] = {n: torch.hann_window(n) for n in stft_fft_sizes}

    def rec_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        """L1 reconstruction loss."""
        return (x - x_hat).abs().mean()

    def stft_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        """Multi-scale STFT spectral loss.

        Computes spectral convergence and log-magnitude L1 across
        multiple FFT sizes.

        Parameters
        ----------
        x, x_hat : Tensor (B, 1, T) or (B, T)
            Audio waveforms.

        Returns
        -------
        Tensor
            Mean spectral loss across FFT sizes.
        """
        # Squeeze channel dim if present: (B, 1, T) -> (B, T)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x_hat.dim() == 3 and x_hat.shape[1] == 1:
            x_hat = x_hat.squeeze(1)

        total_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for n_fft in self.stft_fft_sizes:
            hop = n_fft // 4
            window = self._windows[n_fft].to(device=x.device, dtype=x.dtype)
            spec_x = torch.stft(
                x,
                n_fft,
                hop_length=hop,
                return_complex=True,
                window=window,
            )
            spec_xhat = torch.stft(
                x_hat,
                n_fft,
                hop_length=hop,
                return_complex=True,
                window=window,
            )

            mag_x = spec_x.abs().clamp(min=1e-7)
            mag_xhat = spec_xhat.abs().clamp(min=1e-7)

            # Spectral convergence: Frobenius norm ratio
            sc = (mag_x - mag_xhat).norm(p="fro") / (mag_x.norm(p="fro") + 1e-7)

            # Log-magnitude L1
            log_l1 = (torch.log(mag_x) - torch.log(mag_xhat)).abs().mean()

            total_loss = total_loss + sc + log_l1

        return total_loss / len(self.stft_fft_sizes)

    def chart_loss(self, A: Tensor, A_hat: Tensor) -> Tensor:
        """Spectral chart consistency: ||e_tilde(A) - e_tilde(A_hat)||^2."""
        e = self.prior.band_energies(A)
        e_tilde = e / (e.sum(dim=-1, keepdim=True) + 1e-8)
        e_hat = self.prior.band_energies(A_hat)
        e_tilde_hat = e_hat / (e_hat.sum(dim=-1, keepdim=True) + 1e-8)
        return (e_tilde - e_tilde_hat).pow(2).mean()

    def smooth_loss(self, A: Tensor) -> Tensor:
        """Off-manifold penalty: ||A(I - U_q U_q^T)||_F^2 / ||A||_F^2."""
        return self.prior.off_manifold_energy(A)

    def kl_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Standard VAE KL (unused for audio, kept for interface compat)."""
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

    def __call__(
        self,
        x: Tensor,
        x_hat: Tensor,
        A: Tensor,
        A_hat: Tensor,
        mu: Tensor | None = None,
        logvar: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute all loss components and the weighted total.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with 'rec', 'stft', 'chart', 'smooth', and 'total'
            keys. Includes 'kl' if mu/logvar are provided.
        """
        losses: dict[str, Tensor] = {}
        losses["rec"] = self.rec_loss(x, x_hat)
        losses["stft"] = self.stft_loss(x, x_hat)
        losses["chart"] = self.chart_loss(A, A_hat)
        losses["smooth"] = self.smooth_loss(A)

        total = (
            self.lambda_rec * losses["rec"]
            + self.lambda_stft * losses["stft"]
            + self.lambda_chart * losses["chart"]
            + self.lambda_smooth * losses["smooth"]
        )

        if mu is not None and logvar is not None:
            losses["kl"] = self.kl_loss(mu, logvar)
            total = total + self.lambda_kl * losses["kl"]

        losses["total"] = total
        return losses
