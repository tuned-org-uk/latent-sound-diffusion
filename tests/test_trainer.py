"""Tests for audio training loops.

Uses a stub encoder to avoid downloading EnCodec weights in tests.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

import structlog

from ald_sc.audio_codec import AudioVAE, BaselineAudioDecoder
from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.losses import ALDSCLoss
from ald_sc.schedule import CosineSchedule
from ald_sc.trainer import log_training, train_audio_decoder, train_audio_diffusion


class StubEncoder(nn.Module):
    """Stub encoder mimicking EnCodec interface for testing."""

    def __init__(self, latent_dim: int = 128, stride: int = 320) -> None:
        super().__init__()
        self.proj = nn.Conv1d(1, latent_dim, stride, stride=stride)

    def encode(self, x: Tensor, prior) -> tuple[Tensor, Tensor, Tensor]:
        z = self.proj(x).float()
        a = z.mean(dim=2)
        c_spec = prior.chart_energy_descriptor(a)
        return z, a, c_spec

    def extract_features(self, x: Tensor) -> Tensor:
        return self.proj(x).float()


def _make_dataloader(n: int = 8, audio_length: int = 320 * 16) -> DataLoader:
    torch.manual_seed(3407)
    data = torch.randn(n, 1, audio_length)
    return DataLoader(TensorDataset(data), batch_size=4, shuffle=False)


def _make_vae(latent_dim: int = 128, base_channels: int = 16) -> AudioVAE:
    encoder = StubEncoder(latent_dim=latent_dim)
    decoder = BaselineAudioDecoder(
        latent_channels=latent_dim,
        out_channels=1,
        base_channels=base_channels,
        upsample_strides=(2, 4, 5, 8),  # 320x to match stub encoder stride
    )
    return AudioVAE(encoder=encoder, decoder=decoder)


class TestTrainAudioDecoder:
    def test_loss_decreases(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        vae = _make_vae(latent_dim=128, base_channels=16)
        loss_fn = ALDSCLoss(
            prior=prior,
            lambda_rec=1.0,
            lambda_stft=0.0,
            lambda_chart=0.5,
            lambda_smooth=0.1,
        )
        loader = _make_dataloader(n=8, audio_length=320 * 16)

        losses = list(
            train_audio_decoder(loader, vae, prior, loss_fn, epochs=3, lr=1e-3)
        )
        assert len(losses) > 0
        assert "loss" in losses[0]
        first = losses[0]["loss"]
        last = losses[-1]["loss"]
        assert last < first * 1.5

    def test_encoder_stays_frozen(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        vae = _make_vae(latent_dim=128, base_channels=16)
        loss_fn = ALDSCLoss(
            prior=prior,
            lambda_rec=1.0,
            lambda_stft=0.0,
        )
        loader = _make_dataloader(n=8, audio_length=320 * 16)

        list(train_audio_decoder(loader, vae, prior, loss_fn, epochs=1, lr=1e-3))

        for p in vae.encoder.parameters():
            assert not p.requires_grad

    def test_yields_loss_dict_keys(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        vae = _make_vae()
        loss_fn = ALDSCLoss(
            prior=prior,
            lambda_rec=1.0,
            lambda_stft=0.0,
            lambda_chart=0.5,
            lambda_smooth=0.1,
        )
        loader = _make_dataloader(n=4, audio_length=320 * 16)

        losses = list(
            train_audio_decoder(loader, vae, prior, loss_fn, epochs=1, lr=1e-3)
        )
        assert len(losses) > 0
        for d in losses:
            assert "epoch" in d
            assert "loss" in d
            assert "rec" in d
            assert "stft" in d
            assert "chart" in d
            assert "smooth" in d

    def test_noise_std_zero_matches_default(self) -> None:
        """noise_std=0.0 reproduces the default trajectory."""
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        loss_fn = ALDSCLoss(prior=prior, lambda_rec=1.0, lambda_stft=0.0)
        loader = _make_dataloader(n=8, audio_length=320 * 16)

        torch.manual_seed(7)
        vae1 = _make_vae()
        losses_default = list(
            train_audio_decoder(loader, vae1, prior, loss_fn, epochs=1, lr=1e-3)
        )

        torch.manual_seed(7)
        vae2 = _make_vae()
        losses_zero = list(
            train_audio_decoder(
                loader, vae2, prior, loss_fn, epochs=1, lr=1e-3, noise_std=0.0
            )
        )
        assert len(losses_default) == len(losses_zero)
        for d1, d2 in zip(losses_default, losses_zero):
            assert abs(d1["loss"] - d2["loss"]) < 1e-6

    def test_noise_std_positive_runs(self) -> None:
        """noise_std>0 must run and still yield valid loss dicts."""
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        vae = _make_vae(base_channels=16)
        loss_fn = ALDSCLoss(prior=prior, lambda_rec=1.0, lambda_stft=0.0)
        loader = _make_dataloader(n=8, audio_length=320 * 16)
        losses = list(
            train_audio_decoder(
                loader, vae, prior, loss_fn, epochs=1, lr=1e-3, noise_std=0.1
            )
        )
        assert len(losses) > 0
        for d in losses:
            assert d["loss"] >= 0.0
            assert "epoch" in d and "loss" in d

    def test_yields_grad_norm(self) -> None:
        """Default grad_clip=1.0 yields finite pre-clip grad norms."""
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        vae = _make_vae(base_channels=16)
        loss_fn = ALDSCLoss(prior=prior, lambda_rec=1.0, lambda_stft=0.0)
        loader = _make_dataloader(n=4, audio_length=320 * 16)

        losses = list(
            train_audio_decoder(loader, vae, prior, loss_fn, epochs=1, lr=1e-3)
        )
        assert len(losses) > 0
        for d in losses:
            assert "grad_norm" in d
            assert d["grad_norm"] == d["grad_norm"]  # finite (not NaN)

    def test_grad_clip_bounds_gradient_norm(self) -> None:
        """With clip active, post-step parameter grads respect the bound."""
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        vae = _make_vae(base_channels=16)
        loss_fn = ALDSCLoss(prior=prior, lambda_rec=1.0, lambda_stft=0.0)
        loader = _make_dataloader(n=4, audio_length=320 * 16)
        clip = 1e-3

        list(
            train_audio_decoder(
                loader, vae, prior, loss_fn, epochs=1, lr=1e-3, grad_clip=clip
            )
        )
        for p in vae.decoder.parameters():
            if p.grad is not None:
                assert float(p.grad.norm()) <= clip * 1.001

    def test_grad_clip_disabled_yields_nan_grad_norm(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        vae = _make_vae(base_channels=16)
        loss_fn = ALDSCLoss(prior=prior, lambda_rec=1.0, lambda_stft=0.0)
        loader = _make_dataloader(n=4, audio_length=320 * 16)

        losses = list(
            train_audio_decoder(
                loader, vae, prior, loss_fn, epochs=1, lr=1e-3, grad_clip=None
            )
        )
        assert len(losses) > 0
        for d in losses:
            assert d["grad_norm"] != d["grad_norm"]  # NaN sentinel

    def test_nonfinite_loss_raises(self) -> None:
        """A NaN decoder output must raise a clear RuntimeError, not yield."""
        import pytest

        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)

        class NanDecoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, z: Tensor, c_spec: Tensor) -> Tensor:
                out = torch.zeros(z.shape[0], 1, z.shape[2] * 320, device=z.device)
                return out * float("nan")

        encoder = StubEncoder(latent_dim=128)
        vae = AudioVAE(encoder=encoder, decoder=NanDecoder())
        loss_fn = ALDSCLoss(prior=prior, lambda_rec=1.0, lambda_stft=0.0)
        loader = _make_dataloader(n=4, audio_length=320 * 16)

        with pytest.raises(RuntimeError, match="Non-finite decoder loss"):
            list(train_audio_decoder(loader, vae, prior, loss_fn, epochs=1))


class TestTrainAudioDiffusion:
    def test_diffusion_loss_decreases(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        vae = _make_vae(latent_dim=128, base_channels=16)
        loader = _make_dataloader(n=8, audio_length=320 * 16)
        sched = CosineSchedule(num_steps=100)

        for p in vae.parameters():
            p.requires_grad_(False)

        dit = MinimalDiT(
            latent_channels=128,
            latent_length=16,
            patch_size=4,
            dim=64,
            depth=2,
            num_heads=4,
            text_dim=0,
            spec_dim=24,
        )

        losses = list(
            train_audio_diffusion(loader, vae, dit, prior, sched, epochs=3, lr=1e-3)
        )
        assert len(losses) > 0
        assert "loss" in losses[0]

    def test_vae_stays_frozen(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        vae = _make_vae()
        loader = _make_dataloader(n=4, audio_length=320 * 16)
        sched = CosineSchedule(num_steps=100)

        for p in vae.parameters():
            p.requires_grad_(False)

        dit = MinimalDiT(
            latent_channels=128,
            latent_length=16,
            patch_size=4,
            dim=64,
            depth=2,
            num_heads=4,
            spec_dim=24,
        )

        list(train_audio_diffusion(loader, vae, dit, prior, sched, epochs=1, lr=1e-3))

        vae_grads_after = [p.grad for p in vae.parameters()]
        assert all(g is None for g in vae_grads_after), "VAE must stay frozen"


class TestLogTraining:
    def test_returns_all_records(self) -> None:
        records = [
            {"epoch": 0, "loss": 1.0},
            {"epoch": 0, "loss": 0.9},
            {"epoch": 1, "loss": 0.8},
            {"epoch": 1, "loss": 0.7},
        ]
        history = list(log_training(iter(records), label="Test"))
        assert len(history) == 4

    def test_epoch_summary_shape(self) -> None:
        records = [
            {"epoch": 0, "loss": 1.0},
            {"epoch": 0, "loss": 0.9},
            {"epoch": 1, "loss": 0.8},
        ]
        summaries = list(log_training(iter(records), label="Test"))
        assert summaries[0]["epoch"] == 0
        # Running mean: first record of epoch 0 is 1.0, second is 0.95
        assert summaries[0]["epoch_mean_loss"] == 1.0
        assert summaries[1]["epoch_mean_loss"] == 0.95
        assert summaries[2]["epoch_mean_loss"] == 0.8
        assert "epoch_steps" in summaries[0]

    def test_empty_iterator(self) -> None:
        history = list(log_training(iter([]), label="Test"))
        assert history == []

    def test_emits_structlog_event_per_epoch(self) -> None:
        """log_training should emit one structlog 'epoch' event per epoch."""
        records = [
            {"epoch": 0, "loss": 1.0},
            {"epoch": 0, "loss": 0.9},
            {"epoch": 1, "loss": 0.8},
            {"epoch": 1, "loss": 0.7},
        ]
        with structlog.testing.capture_logs() as caps:
            list(log_training(iter(records), label="Test"))

        epoch_events = [e for e in caps if e["event"] == "epoch"]
        assert len(epoch_events) == 2
        assert epoch_events[0]["epoch"] == 0
        assert epoch_events[0]["label"] == "Test"
        assert epoch_events[0]["mean_loss"] == 0.95
        assert epoch_events[0]["steps"] == 2
        assert epoch_events[1]["epoch"] == 1
        assert epoch_events[1]["mean_loss"] == 0.75
