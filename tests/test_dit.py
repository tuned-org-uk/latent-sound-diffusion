"""Tests for the minimal 1-D DiT denoiser."""

from __future__ import annotations

import torch

from ald_sc.dit import MinimalDiT


def test_forward_shape_with_spectral_conditioning() -> None:
    torch.manual_seed(3407)
    model = MinimalDiT(
        latent_channels=4,
        latent_length=32,
        patch_size=2,
        dim=64,
        depth=2,
        num_heads=4,
        text_dim=0,
        spec_dim=24,
    )
    z = torch.randn(2, 4, 32)
    t = torch.randint(0, 1000, (2,))
    c_spec = torch.randn(2, 24)

    out = model(z, t, c_spec=c_spec)

    assert out.shape == z.shape


def test_forward_shape_with_text_embeddings() -> None:
    torch.manual_seed(3407)
    model = MinimalDiT(
        latent_channels=3,
        latent_length=16,
        patch_size=2,
        dim=32,
        depth=1,
        num_heads=4,
        text_dim=16,
        spec_dim=12,
    )
    z = torch.randn(1, 3, 16)
    t = torch.tensor([500])
    text_emb = torch.randn(1, 16)
    c_spec = torch.randn(1, 12)

    out = model(z, t, text_emb=text_emb, c_spec=c_spec)

    assert out.shape == z.shape


def test_forward_shape_unconditional() -> None:
    """Unconditional forward (no c_spec, no text) should work."""
    torch.manual_seed(3407)
    model = MinimalDiT(
        latent_channels=4,
        latent_length=32,
        patch_size=4,
        dim=64,
        depth=2,
        num_heads=4,
    )
    z = torch.randn(2, 4, 32)
    t = torch.randint(0, 1000, (2,))

    out = model(z, t)

    assert out.shape == z.shape


def test_latent_shape_attribute() -> None:
    """The latent_shape attribute should be set for samplers."""
    model = MinimalDiT(
        latent_channels=128,
        latent_length=375,
        patch_size=8,
        dim=256,
    )
    assert model.latent_shape == (128, 375)


def test_gradient_flow() -> None:
    torch.manual_seed(3407)
    model = MinimalDiT(
        latent_channels=4,
        latent_length=32,
        patch_size=2,
        dim=64,
        depth=2,
        num_heads=4,
        text_dim=0,
        spec_dim=24,
    )
    z = torch.randn(2, 4, 32, requires_grad=True)
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


def test_cfg_dropout_at_full_prob_zeros_c_spec() -> None:
    """At cfg_dropout=1.0, all c_spec should be dropped (zeroed)."""
    torch.manual_seed(3407)
    model = MinimalDiT(
        latent_channels=4,
        latent_length=32,
        patch_size=2,
        dim=64,
        depth=2,
        num_heads=4,
        text_dim=0,
        spec_dim=24,
        cfg_dropout=1.0,
    )
    model.train()
    z = torch.randn(2, 4, 32)
    t = torch.randint(0, 1000, (2,))
    c_spec = torch.randn(2, 24)

    out_dropped = model(z, t, c_spec=c_spec)
    out_uncond = model(z, t, c_spec=torch.zeros(2, 24))
    assert torch.allclose(out_dropped, out_uncond, atol=1e-6)


def test_cfg_dropout_at_zero_prob_uses_c_spec() -> None:
    """At cfg_dropout=0.0, c_spec is always used (when AdaLN is non-zero)."""
    torch.manual_seed(3407)
    model = MinimalDiT(
        latent_channels=4,
        latent_length=32,
        patch_size=2,
        dim=64,
        depth=2,
        num_heads=4,
        text_dim=0,
        spec_dim=24,
        cfg_dropout=0.0,
    )
    # AdaLN is zero-initialized by design; perturb so conditioning is active
    for block in model.blocks:
        torch.nn.init.normal_(block.adaln.proj.weight, std=0.02)
        torch.nn.init.normal_(block.adaln.proj.bias, std=0.02)
    model.train()
    z = torch.randn(2, 4, 32)
    t = torch.randint(0, 1000, (2,))
    c_spec1 = torch.randn(2, 24)
    c_spec2 = torch.randn(2, 24)

    out1 = model(z, t, c_spec=c_spec1)
    out2 = model(z, t, c_spec=c_spec2)
    assert not torch.allclose(out1, out2, atol=1e-6), (
        "Different c_spec must produce different outputs"
    )


def test_conditioning_sensitivity() -> None:
    """Changing c_spec must measurably change the predicted velocity."""
    torch.manual_seed(3407)
    model = MinimalDiT(
        latent_channels=4,
        latent_length=32,
        patch_size=2,
        dim=64,
        depth=2,
        num_heads=4,
        text_dim=0,
        spec_dim=24,
        cfg_dropout=0.0,
    )
    # AdaLN is zero-initialized by design; perturb so conditioning is active
    for block in model.blocks:
        torch.nn.init.normal_(block.adaln.proj.weight, std=0.02)
        torch.nn.init.normal_(block.adaln.proj.bias, std=0.02)
    model.eval()
    z = torch.randn(4, 4, 32)
    t = torch.tensor([100, 200, 300, 400])
    c_spec_a = torch.randn(4, 24)
    c_spec_b = torch.randn(4, 24)

    out_a = model(z, t, c_spec=c_spec_a)
    out_b = model(z, t, c_spec=c_spec_b)
    diff = (out_a - out_b).abs().mean()
    assert diff > 1e-4, f"c_spec must affect output, diff={diff}"
