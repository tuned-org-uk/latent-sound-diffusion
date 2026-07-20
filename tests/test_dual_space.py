"""Tests for the DualSpaceMatrix M_N.

M_N = α ‖V V^T‖_F − β ‖V L_F V^T‖_F

This is the 2.5-D encoding target: the structure defined by the projection
of the item-space into the feature-space graph Laplacian.
"""

from __future__ import annotations

import torch

from ald_sc.dual_space import DualSpaceMatrix
from ald_sc.wire_graph import WireGraph


def _make_graph(n: int = 50, f: int = 16) -> tuple[torch.Tensor, WireGraph]:
    torch.manual_seed(3407)
    features = torch.randn(n, f)
    g = WireGraph.build(features, k=4, seed=3407, mode="feature")
    return features, g


class TestDualSpaceMatrix:
    def test_build_shape(self) -> None:
        features, g = _make_graph(n=50, f=16)
        dsm = DualSpaceMatrix(alpha=0.5, beta=0.5)
        dsm.build(features, g.laplacian_dense)
        assert dsm.M.shape == (50, 50)

    def test_M_is_symmetric(self) -> None:
        features, g = _make_graph(n=50, f=16)
        dsm = DualSpaceMatrix()
        dsm.build(features, g.laplacian_dense)
        assert torch.allclose(dsm.M, dsm.M.T, atol=1e-5)

    def test_eigendecomposition_available(self) -> None:
        features, g = _make_graph(n=50, f=16)
        dsm = DualSpaceMatrix()
        dsm.build(features, g.laplacian_dense)
        assert dsm.eigvals is not None
        assert dsm.eigvecs is not None
        assert dsm.eigvals.shape == (50,)
        assert dsm.eigvecs.shape == (50, 50)

    def test_fiedler_vector_available(self) -> None:
        features, g = _make_graph(n=50, f=16)
        dsm = DualSpaceMatrix()
        dsm.build(features, g.laplacian_dense)
        assert dsm.fiedler_vec is not None
        assert dsm.fiedler_vec.shape == (50,)

    def test_project_shape(self) -> None:
        features, g = _make_graph(n=50, f=16)
        dsm = DualSpaceMatrix()
        dsm.build(features, g.laplacian_dense)
        x = torch.randn(4, 50)
        z = dsm.project(x, k=4)
        assert z.shape == (4, 4)

    def test_rayleigh_shape(self) -> None:
        features, g = _make_graph(n=50, f=16)
        dsm = DualSpaceMatrix()
        dsm.build(features, g.laplacian_dense)
        x = torch.randn(50)
        r = dsm.rayleigh(x)
        assert r.shape == ()

    def test_rayleigh_batch(self) -> None:
        features, g = _make_graph(n=50, f=16)
        dsm = DualSpaceMatrix()
        dsm.build(features, g.laplacian_dense)
        x = torch.randn(4, 50)
        r = dsm.rayleigh(x)
        assert r.shape == (4,)

    def test_no_trainable_parameters(self) -> None:
        """DualSpaceMatrix must be frozen — all cached tensors are buffers."""
        features, g = _make_graph(n=50, f=16)
        dsm = DualSpaceMatrix()
        dsm.build(features, g.laplacian_dense)
        params = list(dsm.parameters())
        assert len(params) == 0

    def test_alpha_beta_affect_M(self) -> None:
        features, g = _make_graph(n=50, f=16)
        dsm1 = DualSpaceMatrix(alpha=1.0, beta=0.0)
        dsm1.build(features, g.laplacian_dense)
        dsm2 = DualSpaceMatrix(alpha=0.0, beta=1.0)
        dsm2.build(features, g.laplacian_dense)
        assert not torch.allclose(dsm1.M, dsm2.M)
