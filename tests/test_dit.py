"""Tests for the minimal 1-D DiT denoiser."""

from __future__ import annotations

import pytest
import torch

from ald_sc.dit import MinimalDiT, _interpolate_pos_embed, _unpatchify


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


@pytest.mark.parametrize(
    "batch,channels,patch_size,num_patches",
    [
        (1, 2, 3, 4),
        (2, 3, 4, 5),
        (1, 128, 8, 47),  # production geometry (v0.11: 375 frames)
        (1, 128, 8, 94),  # target geometry (10 s: 750 frames)
    ],
)
def test_unpatchify_places_features_in_true_temporal_order(
    batch: int, channels: int, patch_size: int, num_patches: int
) -> None:
    """Feature j=c*ps+k of token n must land at time n*ps+k.

    Regression test for the v0.11 layout bug where the (ps, N) merge
    order scattered token features to time slots k*N+n, making trained
    checkpoints length-dependent.
    """
    h = torch.zeros(batch, num_patches, channels * patch_size)
    for n in range(num_patches):
        for c in range(channels):
            for k in range(patch_size):
                h[:, n, c * patch_size + k] = (
                    1_000_000 * n + 1_000 * c + k
                )

    out = _unpatchify(h, batch, channels, patch_size)

    assert out.shape == (batch, channels, num_patches * patch_size)
    for n in range(num_patches):
        for c in range(channels):
            for k in range(patch_size):
                expected = 1_000_000 * n + 1_000 * c + k
                actual = out[:, c, n * patch_size + k]
                assert torch.equal(actual, torch.full_like(actual, expected)), (
                    f"token {n} feature (c={c},k={k}) at time "
                    f"{n * patch_size + k}: got {actual[0].item()}"
                )


class TestInterpolatePosEmbed:
    def test_upsampling_preserves_endpoints(self) -> None:
        """align_corners interpolation must pin the first/last trained positions."""
        torch.manual_seed(3407)
        pos = torch.randn(1, 8, 64)

        up = _interpolate_pos_embed(pos, 15)

        assert up.shape == (1, 15, 64)
        assert torch.allclose(up[0, 0], pos[0, 0], atol=1e-6)
        assert torch.allclose(up[0, -1], pos[0, -1], atol=1e-6)

    def test_downsampling_preserves_endpoints(self) -> None:
        torch.manual_seed(3407)
        pos = torch.randn(1, 15, 64)

        down = _interpolate_pos_embed(pos, 8)

        assert down.shape == (1, 8, 64)
        assert torch.allclose(down[0, 0], pos[0, 0], atol=1e-6)
        assert torch.allclose(down[0, -1], pos[0, -1], atol=1e-6)

    def test_same_length_returns_original(self) -> None:
        pos = torch.randn(1, 8, 64)
        assert _interpolate_pos_embed(pos, 8) is pos


class TestVariableLengthForward:
    def _model(self) -> MinimalDiT:
        torch.manual_seed(3407)
        return MinimalDiT(
            latent_channels=4,
            latent_length=16,
            patch_size=2,
            dim=32,
            depth=1,
            num_heads=4,
            spec_dim=12,
        )

    def test_accepts_lengths_other_than_trained(self) -> None:
        model = self._model()
        t = torch.tensor([500])
        for seq_len in (10, 16, 24, 32):
            z = torch.randn(1, 4, seq_len)
            out = model(z, t)
            assert out.shape == (1, 4, seq_len), f"T={seq_len}"

    def test_longer_input_is_deterministic(self) -> None:
        model = self._model().eval()
        z = torch.randn(1, 4, 48)
        t = torch.tensor([500])
        with torch.no_grad():
            out1 = model(z, t)
            out2 = model(z, t)
        assert torch.allclose(out1, out2)

    def test_gradients_flow_through_interpolated_positions(self) -> None:
        model = self._model()
        z = torch.randn(1, 4, 32)
        t = torch.tensor([500])
        loss = model(z, t).square().mean()
        loss.backward()
        assert model.pos_embed.grad is not None
        assert model.pos_embed.grad.abs().sum() > 0
