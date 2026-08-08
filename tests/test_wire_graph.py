"""Tests for the ArrowSpace WireGraph adapter.

Tests both the ArrowSpace bindings path (when available) and the kNN
fallback. The WireGraph is the single surface through which ALD-SC
accesses the feature-space graph Laplacian L_F and the dispersion
network λ_ED.
"""

from __future__ import annotations

import torch
from ald_sc.wire_graph import WireGraph


class TestWireGraph:
    def test_build_feature_mode(self) -> None:
        torch.manual_seed(3407)
        features = torch.randn(64, 32)
        g = WireGraph.build(features, k=4, seed=3407, mode="feature")
        assert g.laplacian_dense.shape == (32, 32)
        assert g.laplacian_sym.shape == (32, 32)
        assert g.features.shape == (64, 32)

    def test_laplacian_is_symmetric(self) -> None:
        torch.manual_seed(3407)
        features = torch.randn(50, 16)
        g = WireGraph.build(features, k=4, seed=3407, mode="feature")
        L = g.laplacian_dense
        assert torch.allclose(L, L.T, atol=1e-5)

    def test_normalized_laplacian_diagonal_positive(self) -> None:
        torch.manual_seed(3407)
        features = torch.randn(50, 16)
        g = WireGraph.build(features, k=4, seed=3407, mode="feature")
        diag = g.laplacian_sym.diagonal()
        assert (diag >= -1e-5).all()

    def test_rayleigh_quotient_shape(self) -> None:
        torch.manual_seed(3407)
        features = torch.randn(64, 32)
        g = WireGraph.build(features, k=4, seed=3407, mode="feature")
        f = torch.randn(32)
        r = g.rayleigh_quotient(f)
        assert r.shape == (32,)

    def test_spectral_embedding_shape(self) -> None:
        torch.manual_seed(3407)
        features = torch.randn(64, 32)
        g = WireGraph.build(features, k=4, seed=3407, mode="feature")
        z = g.spectral_embedding(n_components=4)
        assert z.shape == (32, 4)

    def test_lambdas_available_when_arrowspace_present(self) -> None:
        """If ArrowSpace bindings are available, lambdas should be non-None."""
        torch.manual_seed(3407)
        features = torch.randn(64, 32)
        g = WireGraph.build(features, k=4, seed=3407, mode="feature")
        from ald_sc.wire_graph import _HAS_ASP

        if _HAS_ASP:
            assert g.lambdas is not None
            assert g.lambdas.shape == (64,)

    def test_laplacian_sparse_output(self) -> None:
        torch.manual_seed(3407)
        features = torch.randn(50, 16)
        g = WireGraph.build(features, k=4, seed=3407, mode="feature")
        L_sparse = g.laplacian(normalized=True, sparse=True)
        assert L_sparse.is_sparse

    def test_laplacian_dense_output(self) -> None:
        torch.manual_seed(3407)
        features = torch.randn(50, 16)
        g = WireGraph.build(features, k=4, seed=3407, mode="feature")
        L_dense = g.laplacian(normalized=False, sparse=False)
        assert L_dense.shape == (16, 16)
        assert not L_dense.is_sparse

    def test_build_item_mode(self) -> None:
        torch.manual_seed(3407)
        features = torch.randn(32, 16)
        g = WireGraph.build(features, k=4, seed=3407, mode="item")
        assert g.laplacian_dense.shape == (32, 32)

    def test_reproducible_with_seed(self) -> None:
        torch.manual_seed(3407)
        features = torch.randn(50, 16)
        g1 = WireGraph.build(features, k=4, seed=3407, mode="feature")
        g2 = WireGraph.build(features, k=4, seed=3407, mode="feature")
        assert torch.allclose(g1.laplacian_dense, g2.laplacian_dense)
