"""Tests for the audio codec module.

EnCodec-dependent tests skip gracefully if the model weights are not
available (no network access). Shape tests use a stub encoder to verify
the AudioVAE and BaselineAudioDecoder interfaces without downloading
the real model.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import Tensor, nn

from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.build_prior import build_arrow_prior
from ald_sc.audio_codec import (
    AudioVAE,
    BaselineAudioDecoder,
    EnCodecEncoder,
    extract_encodec_features,
)


def _make_prior(f: int = 32, q: int = 8) -> ArrowSpacePrior:
    torch.manual_seed(3407)
    embeddings = torch.randn(64, f)
    return build_arrow_prior(embeddings, q=q, k=4)


class StubEncoder(nn.Module):
    """Stub encoder that mimics EnCodec's interface for shape tests."""

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Conv1d(1, latent_dim, 320, stride=320)

    def encode(
        self, x: Tensor, prior: ArrowSpacePrior
    ) -> tuple[Tensor, Tensor, Tensor]:
        z = self.proj(x).float()
        a = z.mean(dim=2)
        c_spec = prior.chart_energy_descriptor(a)
        return z, a, c_spec

    def extract_features(self, x: Tensor) -> Tensor:
        return self.proj(x).float()


class TestBaselineAudioDecoder:
    def test_forward_shape(self) -> None:
        decoder = BaselineAudioDecoder(
            latent_channels=32,
            out_channels=1,
            base_channels=16,
            upsample_strides=(2, 2),
        )
        z = torch.randn(2, 32, 16)
        x_hat = decoder(z)
        # 16 * (2*2) = 64
        assert x_hat.shape == (2, 1, 64)

    def test_gradient_flow(self) -> None:
        decoder = BaselineAudioDecoder(
            latent_channels=32,
            out_channels=1,
            base_channels=16,
            upsample_strides=(2, 2),
        )
        z = torch.randn(2, 32, 16, requires_grad=True)
        x_hat = decoder(z)
        x_hat.sum().backward()
        assert z.grad is not None

    def test_no_c_spec_needed(self) -> None:
        """Baseline decoder takes only z (no spectral conditioning)."""
        decoder = BaselineAudioDecoder(
            latent_channels=32,
            out_channels=1,
            base_channels=16,
            upsample_strides=(2, 2),
        )
        z = torch.randn(1, 32, 8)
        x_hat = decoder(z)
        assert x_hat is not None


class TestAudioVAEWithStub:
    def test_forward_shapes(self) -> None:
        prior = _make_prior(f=128, q=8)
        encoder = StubEncoder(latent_dim=128)
        decoder = BaselineAudioDecoder(
            latent_channels=128,
            out_channels=1,
            base_channels=32,
            upsample_strides=(2, 2),
        )
        vae = AudioVAE(encoder=encoder, decoder=decoder)

        x = torch.randn(2, 1, 320 * 16)  # 16 frames of latent
        z, a, c_spec, x_hat = vae(x, prior)

        assert z.shape == (2, 128, 16)
        assert a.shape == (2, 128)
        assert c_spec.shape == (2, 24)  # 3 * q = 3 * 8
        assert x_hat.shape == (2, 1, 16 * 4)  # 16 * (2*2)

    def test_gradient_flow_decoder(self) -> None:
        """Only the decoder should have gradients (encoder is frozen stub)."""
        prior = _make_prior(f=128, q=8)
        encoder = StubEncoder(latent_dim=128)
        decoder = BaselineAudioDecoder(
            latent_channels=128,
            out_channels=1,
            base_channels=32,
            upsample_strides=(2, 2),
        )
        vae = AudioVAE(encoder=encoder, decoder=decoder)

        x = torch.randn(2, 1, 320 * 16)
        z, a, c_spec, x_hat = vae(x, prior)
        x_hat.sum().backward()

        # Decoder should have gradients
        dec_params = [p for p in decoder.parameters() if p.requires_grad]
        assert dec_params
        for p in dec_params:
            assert p.grad is not None


class TestExtractFeatures:
    def test_extract_features_shape(self) -> None:
        """Extract features from a stub encoder."""
        encoder = StubEncoder(latent_dim=128)
        loader = torch.utils.data.DataLoader(torch.randn(8, 1, 320 * 4), batch_size=4)
        features = extract_encodec_features(loader, encoder)  # type: ignore
        assert features.shape == (8, 128)


class TestEnCodecEncoder:
    """Tests that require the real EnCodec model. Skipped if unavailable."""

    @pytest.fixture
    def encoder(self) -> EnCodecEncoder:
        enc = EnCodecEncoder()
        try:
            enc._load_model()
        except Exception:
            pytest.skip("EnCodec model not available (no network or weights)")
        return enc

    def test_encode_shapes(self, encoder: EnCodecEncoder) -> None:
        prior = _make_prior(f=128, q=8)
        x = torch.randn(1, 1, 24000)  # 1 second @ 24kHz
        z, a, c_spec = encoder.encode(x, prior)
        assert z.shape[0] == 1
        assert z.shape[1] == 128  # EnCodec encoder dim
        assert a.shape == (1, 128)
        assert c_spec.shape == (1, 24)

    def test_extract_features(self, encoder: EnCodecEncoder) -> None:
        x = torch.randn(1, 1, 24000)
        z = encoder.extract_features(x)
        assert z.shape[0] == 1
        assert z.shape[1] == 128

    def test_frozen(self, encoder: EnCodecEncoder) -> None:
        """EnCodec encoder must have no trainable parameters."""
        for p in encoder._encodec.parameters():
            assert not p.requires_grad

    def test_load_emits_no_weight_norm_future_warning(self) -> None:
        """Loading the EnCodec model must not surface the deprecated weight_norm warning.

        The third-party ``encodec`` package constructs its conv stack with the
        deprecated ``torch.nn.utils.weight_norm`` hook (emitting a
        ``FutureWarning``). ``EnCodecEncoder._load_model`` suppresses that
        construction-time warning and migrates modules to the new
        parametrization API, so loading stays warning-free and survives the
        eventual removal of the old hook API.
        """
        enc = EnCodecEncoder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                enc._load_model()
            except Exception:
                pytest.skip("EnCodec model not available (no network or weights)")
        msgs = [
            str(w.message)
            for w in caught
            if issubclass(w.category, FutureWarning) and "weight_norm" in str(w.message)
        ]
        assert not msgs, f"unexpected weight_norm FutureWarning(s): {msgs}"

    def test_loaded_model_has_no_deprecated_weight_norm_hooks(self) -> None:
        """After load, no module should carry the deprecated WeightNorm hook."""
        from torch.nn.utils.weight_norm import WeightNorm

        enc = EnCodecEncoder()
        try:
            enc._load_model()
        except Exception:
            pytest.skip("EnCodec model not available (no network or weights)")

        offenders = [
            type(m).__name__
            for m in enc._encodec.modules()
            if any(isinstance(h, WeightNorm) for h in m._forward_pre_hooks.values())
        ]
        assert not offenders, (
            f"modules still carry deprecated weight_norm hook: {offenders}"
        )


class TestLazyCodecDeviceSync:
    """The lazy-loaded codec must follow the caller's device, however the
    encoder was moved (parent-container moves bypass .to() overrides via
    nn.Module._apply). Regression test for the MPS training crash where
    _encodec loaded onto CPU while inputs were on MPS.
    """

    def test_encode_follows_input_device_after_parent_move(self) -> None:
        if not torch.backends.mps.is_available():
            pytest.skip("requires MPS")
        holder = torch.nn.Module()
        enc = EnCodecEncoder()
        holder.child = enc
        holder.to(torch.device("mps"))

        prior = _make_prior(f=128, q=8).to(torch.device("mps"))
        x = torch.randn(1, 1, 24000, device=torch.device("mps"))
        z, _, _ = enc.encode(x, prior)
        assert z.device.type == "mps"

    def test_extract_features_follows_input_device_after_parent_move(self) -> None:
        if not torch.backends.mps.is_available():
            pytest.skip("requires MPS")
        holder = torch.nn.Module()
        enc = EnCodecEncoder()
        holder.child = enc
        holder.to(torch.device("mps"))

        x = torch.randn(1, 1, 24000, device=torch.device("mps"))
        z = enc.extract_features(x)
        assert z.device.type == "mps"
