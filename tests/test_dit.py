"""Tests for the minimal DiT denoiser."""

from __future__ import annotations

import torch

from ald_sc.dit import MinimalDiT


def test_forward_shape_with_spectral_conditioning() -> None:
    torch.manual_seed(3407)
    model = MinimalDiT(
        latent_channels=4,
        latent_size=32,
        patch_size=2,
        dim=64,
        depth=2,
        num_heads=4,
        text_dim=0,
        spec_dim=24,
    )
    z = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2,))
    c_spec = torch.randn(2, 24)

    out = model(z, t, c_spec=c_spec)

    assert out.shape == z.shape


def test_forward_shape_with_text_embeddings() -> None:
    torch.manual_seed(3407)
    model = MinimalDiT(
        latent_channels=3,
        latent_size=16,
        patch_size=2,
        dim=32,
        depth=1,
        num_heads=4,
        text_dim=16,
        spec_dim=12,
    )
    z = torch.randn(1, 3, 16, 16)
    t = torch.tensor([500])
    text_emb = torch.randn(1, 16)
    c_spec = torch.randn(1, 12)

    out = model(z, t, text_emb=text_emb, c_spec=c_spec)

    assert out.shape == z.shape


def test_gradient_flow() -> None:
    torch.manual_seed(3407)
    model = MinimalDiT(
        latent_channels=4,
        latent_size=32,
        patch_size=2,
        dim=64,
        depth=2,
        num_heads=4,
        text_dim=0,
        spec_dim=24,
    )
    z = torch.randn(2, 4, 32, 32, requires_grad=True)
    t = torch.randint(0, 1000, (2,))
    c_spec = torch.randn(2, 24)

    out = model(z, t, c_spec=c_spec)
    loss = out.square().mean()
    loss.backward()

    assert out.shape == z.shape
    assert z.grad is not None
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable
    for p in trainable:
        assert p.grad is not None, f"{type(p).__name__} has no gradient"
