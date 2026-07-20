"""Tests for the frozen ArrowSpace prior and spectral chart construction."""

from __future__ import annotations

import torch

from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.build_prior import build_arrow_prior


def _make_toy_prior(f: int = 32, q: int = 8) -> ArrowSpacePrior:
    torch.manual_seed(3407)
    embeddings = torch.randn(64, f)
    return build_arrow_prior(embeddings, q=q, k=4)


class TestArrowSpacePrior:
    def test_construction_shapes(self) -> None:
        prior = _make_toy_prior(f=32, q=8)
        assert prior.L_F.shape == (32, 32)
        assert prior.U_q.shape == (32, 8)
        assert prior.eigvals_q.shape == (8,)
        assert prior.lambdas_ed.shape == (32,)
        assert prior.lambdas_chart.shape == (8,)

    def test_frozen_no_parameters(self) -> None:
        prior = _make_toy_prior()
        params = list(prior.parameters())
        assert len(params) == 0, "ArrowSpacePrior must have zero trainable parameters"

    def test_buffers_exist(self) -> None:
        prior = _make_toy_prior()
        buffers = list(prior.buffers())
        assert len(buffers) >= 4, (
            "Must have at least L_F, U_q, eigvals_q, lambdas_ed as buffers"
        )

    def test_project_to_chart_idempotent(self) -> None:
        """Pi^2 = Pi: projecting twice should equal projecting once."""
        prior = _make_toy_prior(f=32, q=8)
        A = torch.randn(4, 32)
        A_proj1 = prior.project_to_chart(A)
        A_proj2 = prior.project_to_chart(A_proj1)
        assert torch.allclose(A_proj1, A_proj2, atol=1e-5)

    def test_off_manifold_energy_in_range(self) -> None:
        prior = _make_toy_prior(f=32, q=8)
        A = torch.randn(4, 32)
        energy = prior.off_manifold_energy(A)
        assert energy >= 0.0 - 1e-6
        assert energy <= 1.0 + 1e-6

    def test_off_manifold_energy_zero_on_subspace(self) -> None:
        prior = _make_toy_prior(f=32, q=8)
        A = torch.randn(4, 32)
        A_proj = prior.project_to_chart(A)
        energy = prior.off_manifold_energy(A_proj)
        assert energy < 1e-5, "Energy of projected signal should be near zero"

    def test_chart_coefficients_shape(self) -> None:
        prior = _make_toy_prior(f=32, q=8)
        A = torch.randn(4, 32)
        s = prior.chart_coefficients(A)
        assert s.shape == (4, 8)

    def test_band_energies_shape(self) -> None:
        prior = _make_toy_prior(f=32, q=8)
        A = torch.randn(4, 32)
        e = prior.band_energies(A)
        assert e.shape == (4, 8)
        assert (e >= 0).all()

    def test_chart_energy_descriptor_shape(self) -> None:
        prior = _make_toy_prior(f=32, q=8)
        A = torch.randn(4, 32)
        c_spec = prior.chart_energy_descriptor(A)
        assert c_spec.shape == (4, 24), "c_spec should be 3*q = 24"

    def test_chart_energy_descriptor_components(self) -> None:
        prior = _make_toy_prior(f=32, q=8)
        A = torch.randn(4, 32)
        c_spec = prior.chart_energy_descriptor(A)
        e_tilde = c_spec[:, :8]
        lambda_chart = c_spec[:, 8:16]
        nu = c_spec[:, 16:24]
        assert (e_tilde >= 0).all()
        assert (lambda_chart >= 0).all()
        assert (nu >= 0).all()


class TestBuildPrior:
    def test_build_from_embeddings(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(100, 16)
        prior = build_arrow_prior(embeddings, q=4, k=4)
        assert prior.L_F.shape == (16, 16)
        assert prior.U_q.shape == (16, 4)

    def test_laplacian_symmetric(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(50, 16)
        prior = build_arrow_prior(embeddings, q=4, k=4)
        L = prior.L_F
        assert torch.allclose(L, L.T, atol=1e-5)

    def test_eigenvalues_sorted_ascending(self) -> None:
        prior = _make_toy_prior(f=32, q=8)
        eigvals = prior.eigvals_q
        diffs = eigvals[1:] - eigvals[:-1]
        assert (diffs >= -1e-5).all(), "Eigenvalues should be sorted ascending"
