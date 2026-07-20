"""ALD-SC loss components.

Computes the four spectral loss terms from the design doc:
- L_rec: reconstruction (L1 + optional perceptual)
- L_chart: spectral chart consistency
- L_smooth: off-manifold penalty
- L_kl: VAE latent KL (added for paper consistency)

This module must not import trainer or model modules except for type hints
(per AGENTS.md §11).
"""

from __future__ import annotations

from torch import Tensor

from ald_sc.arrow_prior import ArrowSpacePrior

__all__ = ["ALDSCLoss"]


class ALDSCLoss:
    """Composite loss for ALD-SC spectral VAE training.

    Parameters
    ----------
    prior : ArrowSpacePrior
        The frozen ArrowSpace prior for spectral computations.
    lambda_rec : float
        Weight for reconstruction loss.
    lambda_chart : float
        Weight for spectral chart consistency loss.
    lambda_smooth : float
        Weight for off-manifold penalty.
    lambda_kl : float
        Weight for VAE KL regularisation.
    """

    def __init__(
        self,
        prior: ArrowSpacePrior,
        lambda_rec: float = 1.0,
        lambda_chart: float = 0.5,
        lambda_smooth: float = 0.1,
        lambda_kl: float = 0.0,
    ) -> None:
        self.prior = prior
        self.lambda_rec = lambda_rec
        self.lambda_chart = lambda_chart
        self.lambda_smooth = lambda_smooth
        self.lambda_kl = lambda_kl

    def rec_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        """L1 reconstruction loss."""
        return (x - x_hat).abs().mean()

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
        """Standard VAE KL: 0.5 * (mu^2 + sigma^2 - log(sigma^2) - 1)."""
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
            Dictionary with 'rec', 'chart', 'smooth', and 'total' keys.
            Includes 'kl' if mu/logvar are provided.
        """
        losses: dict[str, Tensor] = {}
        losses["rec"] = self.rec_loss(x, x_hat)
        losses["chart"] = self.chart_loss(A, A_hat)
        losses["smooth"] = self.smooth_loss(A)

        total = (
            self.lambda_rec * losses["rec"]
            + self.lambda_chart * losses["chart"]
            + self.lambda_smooth * losses["smooth"]
        )

        if mu is not None and logvar is not None:
            losses["kl"] = self.kl_loss(mu, logvar)
            total = total + self.lambda_kl * losses["kl"]

        losses["total"] = total
        return losses
