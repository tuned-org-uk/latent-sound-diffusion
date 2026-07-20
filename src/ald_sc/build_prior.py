"""Build the frozen ArrowSpace prior from corpus embeddings.

Constructs the feature-space graph Laplacian via kNN over feature columns
(the ArrowSpace dual-space construction), eigendecomposes it, and returns
an ``ArrowSpacePrior`` with frozen buffers.

This module must not import ``torch.nn`` modules (per AGENTS.md §11).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ald_sc.arrow_prior import ArrowSpacePrior

__all__ = ["build_arrow_prior", "build_feature_laplacian", "build_projector"]


def build_feature_laplacian(embeddings: Tensor, k: int = 8) -> tuple[Tensor, Tensor]:
    """Build the feature-space graph Laplacian from a corpus of embeddings.

    Features are columns of the embedding matrix. The kNN graph is built on
    cosine similarity between feature signals (columns), exactly as in
    ArrowSpace's feature-Laplacian construction.

    Parameters
    ----------
    embeddings : Tensor (N, F)
        Corpus of N embeddings with F features each.
    k : int
        Number of nearest neighbours per feature.

    Returns
    -------
    L_F : Tensor (F, F)
        Unnormalised feature-space Laplacian (D - W).
    W : Tensor (F, F)
        Symmetrised adjacency matrix.
    """
    X = embeddings.detach().float()
    N, F = X.shape

    X_feat = X.T
    norms = X_feat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    X_norm = X_feat / norms
    S = X_norm @ X_norm.T

    W = torch.zeros_like(S)
    for i in range(F):
        nbrs = torch.argsort(-S[i])[1 : k + 1]
        W[i, nbrs] = S[i, nbrs]
    W = torch.maximum(W, W.T)

    D = torch.diag(W.sum(dim=1))
    L = D - W
    return L, W


def build_projector(
    L_F: Tensor, q: int, drop_constant: bool = True
) -> tuple[Tensor, Tensor]:
    """Compute the leading q eigenvectors of the Laplacian.

    Parameters
    ----------
    L_F : Tensor (F, F)
        Feature-space Laplacian.
    q : int
        Number of smooth modes to retain.
    drop_constant : bool
        If True, skip the first (trivial constant) eigenvector.

    Returns
    -------
    U_q : Tensor (F, q)
        Leading q eigenvectors.
    eigvals_q : Tensor (q,)
        Corresponding eigenvalues.
    """
    eigvals, eigvecs = torch.linalg.eigh(L_F)
    idx = torch.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    start = 1 if drop_constant else 0
    q = min(q, eigvecs.shape[1] - start)
    U_q = eigvecs[:, start : start + q]
    eigvals_q = eigvals[start : start + q]
    return U_q, eigvals_q


def _energy_dispersion(embeddings: Tensor) -> Tensor:
    """Compute the ArrowSpace energy-dispersion distribution per feature.

    Normalised per-feature energy relative to the total.

    Parameters
    ----------
    embeddings : Tensor (N, F)

    Returns
    -------
    Tensor (F,) in [0, 1]
    """
    X = embeddings.detach().float()
    energy = X.pow(2).sum(dim=0)
    total = energy.sum() + 1e-12
    return energy / total


def build_arrow_prior(
    embeddings: Tensor,
    q: int = 16,
    k: int = 8,
) -> ArrowSpacePrior:
    """Build a frozen ArrowSpacePrior from corpus embeddings.

    Parameters
    ----------
    embeddings : Tensor (N, F)
        Corpus of N embeddings with F features.
    q : int
        Number of smooth spectral modes to retain.
    k : int
        kNN neighbours per feature for graph construction.

    Returns
    -------
    ArrowSpacePrior
        Frozen prior with buffers (no trainable parameters).
    """
    L_F, _ = build_feature_laplacian(embeddings, k=k)
    U_q, eigvals_q = build_projector(L_F, q=q)
    lambdas_ed = _energy_dispersion(embeddings)
    return ArrowSpacePrior(L_F=L_F, U_q=U_q, eigvals_q=eigvals_q, lambdas_ed=lambdas_ed)
