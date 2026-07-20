"""Tests for ALD-SC loss components."""

from __future__ import annotations

import torch

from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.build_prior import build_arrow_prior
from ald_sc.losses import ALDSCLoss


def _make_prior(f: int = 32, q: int = 8) -> ArrowSpacePrior:
    torch.manual_seed(3407)
    embeddings = torch.randn(64, f)
    return build_arrow_prior(embeddings, q=q, k=4)


class TestALDSCLoss:
    def test_rec_loss_shape(self) -> None:
        prior = _make_prior()
        loss_fn = ALDSCLoss(prior=prior)
        x = torch.randn(2, 3, 32, 32)
        x_hat = torch.randn(2, 3, 32, 32)
        A = torch.randn(2, 32)
        A_hat = torch.randn(2, 32)
        losses = loss_fn(x, x_hat, A, A_hat)
        assert "rec" in losses
        assert losses["rec"].dim() == 0

    def test_chart_loss_shape(self) -> None:
        prior = _make_prior()
        loss_fn = ALDSCLoss(prior=prior)
        A = torch.randn(2, 32)
        A_hat = torch.randn(2, 32)
        chart_loss = loss_fn.chart_loss(A, A_hat)
        assert chart_loss.dim() == 0
        assert chart_loss >= 0

    def test_smooth_loss_shape(self) -> None:
        prior = _make_prior()
        loss_fn = ALDSCLoss(prior=prior)
        A = torch.randn(2, 32)
        smooth_loss = loss_fn.smooth_loss(A)
        assert smooth_loss.dim() == 0
        assert smooth_loss >= 0

    def test_smooth_loss_zero_on_subspace(self) -> None:
        prior = _make_prior()
        loss_fn = ALDSCLoss(prior=prior)
        A = torch.randn(2, 32)
        A_proj = prior.project_to_chart(A)
        smooth_loss = loss_fn.smooth_loss(A_proj)
        assert smooth_loss < 1e-5

    def test_total_loss_has_components(self) -> None:
        prior = _make_prior()
        loss_fn = ALDSCLoss(
            prior=prior, lambda_rec=1.0, lambda_chart=0.5, lambda_smooth=0.1
        )
        x = torch.randn(2, 3, 32, 32)
        x_hat = torch.randn(2, 3, 32, 32)
        A = torch.randn(2, 32)
        A_hat = torch.randn(2, 32)
        losses = loss_fn(x, x_hat, A, A_hat)
        assert "rec" in losses
        assert "chart" in losses
        assert "smooth" in losses
        assert "total" in losses

    def test_gradient_flow(self) -> None:
        prior = _make_prior()
        loss_fn = ALDSCLoss(prior=prior)
        x = torch.randn(2, 3, 16, 16)
        x_hat = torch.randn(2, 3, 16, 16, requires_grad=True)
        A = torch.randn(2, 32)
        A_hat = torch.randn(2, 32, requires_grad=True)
        losses = loss_fn(x, x_hat, A, A_hat)
        losses["total"].backward()
        assert x_hat.grad is not None
        assert A_hat.grad is not None
