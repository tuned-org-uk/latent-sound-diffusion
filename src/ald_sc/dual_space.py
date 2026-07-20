"""DualSpaceMatrix: the 2.5-D encoding target.

M_N = α ‖V V^T‖_F − β ‖V L_F V^T‖_F

where V ∈ R^{N×F} is the corpus embedding matrix and L_F ∈ R^{F×F} is the
feature-space graph Laplacian from ArrowSpace.

This is the structure defined by the projection of the item-space into the
feature-space graph Laplacian. It is the training dataset for encoding:
the encoder learns to produce feature fields A whose projection onto U_q
respects this fused geometric-semantic structure.

Ported from ESDM's ``esdm/graphs/laplacian.py``, adapted for ALD-SC.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["DualSpaceMatrix"]


class DualSpaceMatrix(nn.Module):
    """Builds and caches the fused item-space geometric-semantic matrix M_N.

    Parameters
    ----------
    alpha : float
        Weight on the pure geometric (cosine) term.
    beta : float
        Weight on the spectral-semantic (Laplacian-modulated) term.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.register_buffer("M", None)
        self.register_buffer("eigvals", None)
        self.register_buffer("eigvecs", None)
        self.register_buffer("fiedler_vec", None)

    def build(
        self,
        V: Tensor,
        L_F: Tensor,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        """Compute and cache M_N from corpus embeddings V and L_F.

        Parameters
        ----------
        V : Tensor (N, F)
            Corpus embedding matrix.
        L_F : Tensor (F, F)
            Feature-space graph Laplacian.
        device : torch.device, optional
        """
        if device is None:
            device = V.device

        V = V.to(device=device, dtype=torch.float64)
        L = L_F.to(device=device, dtype=torch.float64)

        N, F = V.shape
        if L.shape != (F, F):
            raise ValueError(
                f"L_F shape {tuple(L.shape)} does not match V feature dim F={F}."
            )

        G = V @ V.T
        S = (V @ L) @ V.T

        M = self.alpha * self._frobenius_norm(G) - self.beta * self._frobenius_norm(S)
        self.M = self._symmetrise(M)

        self.eigvals, self.eigvecs = torch.linalg.eigh(self.M)
        self.fiedler_vec = self.eigvecs[:, 1]

    def project(self, x: Tensor, k: int) -> Tensor:
        """Project item vectors x onto the top-k eigenvectors of M_N.

        Parameters
        ----------
        x : Tensor (N, D) or (D,)
        k : int
            Number of eigenvectors to retain.

        Returns
        -------
        Tensor (N, k)
        """
        if self.eigvecs is None:
            raise RuntimeError("Call build() before project().")
        if k < 1 or k > self.eigvecs.shape[1]:
            raise ValueError(f"k={k} must be in [1, {self.eigvecs.shape[1]}].")

        x = x.to(dtype=self.eigvecs.dtype, device=self.eigvecs.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        top_k_vecs = self.eigvecs[:, -k:]
        return x @ top_k_vecs

    def rayleigh(self, x: Tensor) -> Tensor:
        """Rayleigh quotient R(x) = x^T M x / x^T x.

        Parameters
        ----------
        x : Tensor (N, D) or (D,)

        Returns
        -------
        Tensor — scalar if 1-D, (N,) if 2-D.
        """
        if self.M is None:
            raise RuntimeError("Call build() before rayleigh().")

        squeeze = x.dim() == 1
        x = x.to(dtype=self.M.dtype, device=self.M.device)
        if squeeze:
            x = x.unsqueeze(0)

        Mx = x @ self.M
        numerator = (x * Mx).sum(dim=-1)
        denominator = (x * x).sum(dim=-1).clamp(min=1e-12)
        result = numerator / denominator
        return result.squeeze(0) if squeeze else result

    @staticmethod
    def _frobenius_norm(A: Tensor) -> Tensor:
        return A / (A.norm("fro") + 1e-12)

    @staticmethod
    def _symmetrise(A: Tensor) -> Tensor:
        return (A + A.T) * 0.5
