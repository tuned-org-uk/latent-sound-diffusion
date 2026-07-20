"""Frozen ArrowSpace prior for spectral chart conditioning.

Stores the corpus-level feature-space graph Laplacian and its spectral
decomposition as non-trainable buffers. Provides methods to project feature
fields onto the smooth subspace and extract compact spectral chart
conditioning vectors.

This is the ALD-SC counterpart of ESDM's frozen-prior concept. The prior is
built once from a training corpus and never updated during training (zero
``nn.Parameter``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["ArrowSpacePrior"]


class ArrowSpacePrior(nn.Module):
    """Frozen ArrowSpace spectral prior.

    Stores L_F, U_q, eigvals_q, lambdas_ed, lambdas_chart as buffers.
    Provides projection, off-manifold energy, chart coefficients, band
    energies, and the full c_spec conditioning vector.

    Parameters
    ----------
    L_F : Tensor (F, F)
        Feature-space graph Laplacian.
    U_q : Tensor (F, q)
        Leading q eigenvectors (smooth modes).
    eigvals_q : Tensor (q,)
        Corresponding eigenvalues.
    lambdas_ed : Tensor (F,)
        Energy-dispersion distribution per feature node.
    """

    def __init__(
        self,
        L_F: Tensor,
        U_q: Tensor,
        eigvals_q: Tensor,
        lambdas_ed: Tensor,
    ) -> None:
        super().__init__()
        self.F = L_F.shape[0]
        self.q = U_q.shape[1]

        self.register_buffer("L_F", L_F.float())
        self.register_buffer("U_q", U_q.float())
        self.register_buffer("eigvals_q", eigvals_q.float())
        self.register_buffer("lambdas_ed", lambdas_ed.float())

        lambdas_chart = (lambdas_ed.unsqueeze(1) * U_q.pow(2)).sum(dim=0)
        self.register_buffer("lambdas_chart", lambdas_chart.float())

    def project_to_chart(self, A: Tensor) -> Tensor:
        """Project feature field onto smooth subspace: A @ U_q @ U_q^T.

        Parameters
        ----------
        A : Tensor (B, F) or (..., F)
            Feature field.

        Returns
        -------
        Tensor : same shape as A, projected onto the smooth subspace.
        """
        return A @ self.U_q @ self.U_q.T

    def off_manifold_energy(self, A: Tensor) -> Tensor:
        """Relative Frobenius energy outside the smooth subspace.

        ||A(I - U_q U_q^T)||_F^2 / (||A||_F^2 + eps)

        Returns scalar tensor.
        """
        A_perp = A - self.project_to_chart(A)
        num = A_perp.pow(2).sum()
        den = A.pow(2).sum() + 1e-8
        return num / den

    def chart_coefficients(self, A: Tensor) -> Tensor:
        """Spectral chart coordinates: s = Pool(A) @ U_q.

        Pool is mean over the spatial axis (dim=-2 for (B, N, F) input,
        or identity for (B, F) input).
        """
        if A.dim() == 3:
            A_pooled = A.mean(dim=1)
        else:
            A_pooled = A
        return A_pooled @ self.U_q

    def band_energies(self, A: Tensor) -> Tensor:
        """Per-mode band energies: e_k = ||A u_k||^2 / N.

        Parameters
        ----------
        A : Tensor (B, F) or (B, N, F)

        Returns
        -------
        Tensor (B, q)
        """
        if A.dim() == 3:
            A_pooled = A.mean(dim=1)
        else:
            A_pooled = A
        return (A_pooled @ self.U_q).pow(2) / (
            A_pooled.pow(2).sum(dim=-1, keepdim=True) + 1e-8
        )

    def chart_energy_descriptor(self, A: Tensor) -> Tensor:
        """Full spectral conditioning vector c_spec = [e_tilde, lambda_chart, nu].

        Concatenates:
        - Normalized band energies e_tilde (B, q)
        - Projected energy-dispersion lambda_chart (q,) broadcast to (B, q)
        - Laplacian eigenvalues nu (q,) broadcast to (B, q)

        Returns
        -------
        Tensor (B, 3*q)
        """
        e = self.band_energies(A)
        e_tilde = e / (e.sum(dim=-1, keepdim=True) + 1e-8)

        B = e_tilde.shape[0]
        lambda_chart = self.lambdas_chart.unsqueeze(0).expand(B, -1)
        nu = self.eigvals_q.unsqueeze(0).expand(B, -1)

        return torch.cat([e_tilde, lambda_chart, nu], dim=-1)
