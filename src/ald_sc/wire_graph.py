"""ArrowSpace adapter — the single surface for graph construction.

Wraps the real ``arrowspace`` Rust bindings (pyarrowspace) when available,
falling back to a local kNN Laplacian when they are absent.

This module provides:
- Feature-space graph Laplacian L_F (F×F) from corpus embeddings
- Per-item energy-dispersion scores λ_ED (the ``lambdas()`` method)
- Rayleigh quotient R(f) = f^T L f / f^T f
- Spectral embedding (leading non-trivial eigenvectors)

This is the ALD-SC port of ESDM's ``esdm/graphs/wiring.py``, stripped of
ESDM-specific imports. Per AGENTS.md §1.1, the ArrowSpace library should
be the single source of truth for L_F and λ_ED.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

try:
    import arrowspace as asp  # type: ignore

    _HAS_ASP = True
except Exception:
    asp = None  # type: ignore
    _HAS_ASP = False

__all__ = ["WireGraph", "_HAS_ASP"]


def _normalize_symmetric(L: Tensor) -> Tensor:
    """L_sym = D^{-1/2} L D^{-1/2} (symmetric normalised Laplacian)."""
    d = L.diagonal().clamp(min=0.0)
    d_inv_sqrt = torch.where(d > 0, d.rsqrt(), torch.zeros_like(d))
    D = torch.diag(d_inv_sqrt)
    return D @ L @ D


@dataclass
class WireGraph:
    """Feature-space graph Laplacian built via ArrowSpace.

    The Laplacian is (F, F) where F is the feature dimension of the input
    data. ArrowSpace builds the graph from item co-occurrence structure:
    pass data as (N, F) where N = items, F = features.

    ArrowSpace is the space defined by the graph Laplacian (gl) and the
    energy distribution of the lambdas (lambdas()). Both are stored here
    so the decoder can use the full ArrowSpace structure.

    Attributes
    ----------
    laplacian_dense : Tensor (F, F)
        Unnormalised feature-space Laplacian.
    laplacian_sym : Tensor (F, F)
        Symmetric normalised Laplacian, cached.
    features : Tensor (N, D)
        Source items.
    nnz : int
        Number of non-zero entries in L.
    lambdas : Tensor (N,) or None
        Per-item ArrowSpace energy scores. Available only when the
        ArrowSpace Rust bindings are installed.
    """

    laplacian_dense: Tensor
    laplacian_sym: Tensor
    features: Tensor
    nnz: int
    lambdas: Tensor | None = None

    @classmethod
    def build(
        cls,
        features: Tensor,
        k: int = 4,
        seed: int = 3407,
        mode: str = "feature",
        graph_params: dict | None = None,
    ) -> WireGraph:
        """Build a graph Laplacian via ArrowSpace.

        Parameters
        ----------
        features : Tensor (N, F)
            Item-feature matrix. N items, F features.
        k : int
            Max clusters for ArrowSpace builder (feature mode) or k-NN
            neighbours (item mode). Default 4.
        seed : int
            Random seed.
        mode : str
            "feature" — ArrowSpace returns (F, F) feature-space Laplacian.
            "item" — k-NN Laplacian on item rows, returns (N, N).
        graph_params : dict, optional
            Pre-tuned ArrowSpace parameters (eps, k, tau).
        """
        lambdas = None
        if mode == "feature" and _HAS_ASP:
            items_np = features.detach().cpu().to(torch.float64).numpy()
            L, lambdas = cls._arrowspace_laplacian(
                items_np, k=k, seed=seed, graph_params=graph_params
            )
        else:
            L = cls._knn_laplacian_fallback(features, k=k, seed=seed, mode=mode)
        L = L.to(dtype=torch.float32)
        L_sym = _normalize_symmetric(L)
        return cls(
            laplacian_dense=L,
            laplacian_sym=L_sym,
            features=features,
            nnz=int((L != 0).sum()),
            lambdas=lambdas,
        )

    @staticmethod
    def _arrowspace_laplacian(
        items_np: np.ndarray,
        k: int,
        seed: int,
        graph_params: dict | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Build L_F via the real ArrowSpace Rust bindings.

        Returns (L_F, lambdas) where lambdas are the per-item ArrowSpace
        energy scores.
        """
        gp = dict(graph_params) if graph_params else None
        if gp and gp.get("eps") is not None:
            import arrowspace as asp

            gp.pop("tau", None)
            builder = (
                asp.ArrowSpaceBuilder()
                .with_seed(seed)
                .with_sampling("simple", 1.0)
                .with_dims_reduction(False, None)
            )
            aspace, gl = builder.build_full(graph_params=gp, items=items_np)
            L_np = np.array(gl.to_dense())
            lambdas_np = np.array(aspace.lambdas())
        else:
            import arrowspace_tuner as arrowspace

            aspace, gl = arrowspace.optuna(
                embeddings=items_np,
                seed=seed,
                k_low=max(3, k - 5),
                k_high=k + 5,
            )
            L_np = np.array(gl.to_dense())
            lambdas_np = np.array(aspace.lambdas())

        L = torch.from_numpy(L_np)
        lambdas = (
            torch.from_numpy(lambdas_np.astype(np.float32)) if lambdas_np is not None else None
        )
        return L, lambdas

    @staticmethod
    def _knn_laplacian_fallback(
        features: Tensor, k: int, seed: int, mode: str = "feature"
    ) -> Tensor:
        """Fallback k-NN Laplacian when ArrowSpace bindings are absent.

        In feature mode, builds the graph over feature columns (the
        ArrowSpace dual-space construction). In item mode, builds over
        item rows.
        """
        if mode == "feature":
            data = features.T
        else:
            data = features

        n = data.shape[0]
        f = data / (data.norm(dim=-1, keepdim=True) + 1e-8)
        sims = f @ f.T
        sims.fill_diagonal_(-float("inf"))
        k = min(k, n - 1)
        vals, idx = sims.topk(k, dim=-1)
        weights = torch.clamp(vals, min=0.0)
        A = torch.zeros(n, n)
        A.scatter_(1, idx, weights)
        A = (A + A.T) * 0.5
        D = A.sum(dim=-1)
        L = torch.diag(D) - A
        return L

    def laplacian(self, normalized: bool = True, sparse: bool = True) -> Tensor:
        """Return the Laplacian in the requested format.

        Parameters
        ----------
        normalized : bool
            If True, return the symmetric normalised Laplacian.
        sparse : bool
            If True, return a sparse COO tensor.
        """
        L = self.laplacian_sym if normalized else self.laplacian_dense
        if not sparse:
            return L
        i = L.nonzero(as_tuple=False).t()
        v = L[i[0], i[1]]
        return torch.sparse_coo_tensor(i, v, L.shape).coalesce()

    def rayleigh_quotient(self, f: Tensor) -> Tensor:
        """Per-node Rayleigh energy density: R_k(f) = f_k (L f)_k / (f^T f)."""
        f = f.to(dtype=self.laplacian_dense.dtype, device=self.laplacian_dense.device)
        Lf = self.laplacian_dense @ f
        den = (f * f).sum(dim=-1) + 1e-8
        return f * Lf / den

    def spectral_embedding(self, n_components: int = 2) -> Tensor:
        """Leading non-trivial eigenvectors of the normalised Laplacian."""
        vals, vecs = torch.linalg.eigh(self.laplacian_sym)
        start = 1
        z = vecs[:, start : start + n_components]
        return z
