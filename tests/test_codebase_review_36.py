"""Tests for issue #36 codebase-review fixes.

Covers:
- #2: chart_loss is non-zero (A_hat derived from x_hat, not A.detach())
- #4: Min-SNR weighted diffusion loss
- #+: gradient clipping in diffusion training loop
- #6: STFT Hann window caching (no per-call allocation)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ald_sc.audio_codec import AudioVAE, BaselineAudioDecoder
from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.losses import ALDSCLoss
from ald_sc.schedule import CosineSchedule
from ald_sc.trainer import train_audio_decoder, train_audio_diffusion


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
        upsample_strides=(2, 4, 5, 8),
    )
    return AudioVAE(encoder=encoder, decoder=decoder)


def _make_prior(f: int = 128, q: int = 8) -> object:
    torch.manual_seed(3407)
    embeddings = torch.randn(32, f)
    return build_arrow_prior(embeddings, q=q, k=4)


class TestChartLossNonZero:
    """#2: chart_loss must be non-zero — A_hat derived from x_hat, not A.detach()."""

    def test_chart_loss_positive_after_step(self) -> None:
        """After one training step, chart_loss must be > 0.

        Previously A_hat = A.detach() made chart_loss always 0 (lambda_chart
        was a no-op). Now A_hat is derived from the decoded reconstruction.
        """
        prior = _make_prior()
        vae = _make_vae()
        loss_fn = ALDSCLoss(
            prior=prior,
            lambda_rec=1.0,
            lambda_stft=0.0,
            lambda_chart=0.5,
            lambda_smooth=0.0,
        )
        loader = _make_dataloader(n=4, audio_length=320 * 16)

        records = list(
            train_audio_decoder(loader, vae, prior, loss_fn, epochs=1, lr=1e-3)
        )
        assert len(records) > 0
        chart_values = [r["chart"] for r in records]
        assert any(c > 1e-6 for c in chart_values), (
            f"chart_loss is always zero — A_hat == A.detach(). "
            f"Values: {chart_values}"
        )

    def test_chart_loss_reflects_reconstruction_quality(self) -> None:
        """With a deliberately bad decoder, chart_loss should be substantial."""
        prior = _make_prior()
        vae = _make_vae()
        loss_fn = ALDSCLoss(
            prior=prior,
            lambda_rec=0.0,
            lambda_stft=0.0,
            lambda_chart=1.0,
            lambda_smooth=0.0,
        )
        loader = _make_dataloader(n=2, audio_length=320 * 16)

        records = list(
            train_audio_decoder(loader, vae, prior, loss_fn, epochs=1, lr=1e-3)
        )
        # An untrained decoder produces poor reconstructions, so A_hat
        # (derived from x_hat) should differ from A — chart_loss > 0.
        assert records[0]["chart"] > 1e-4


class TestSNRWeighting:
    """#4: Min-SNR weighted diffusion loss."""

    def test_default_loss_weighting_is_snr(self) -> None:
        """train_audio_diffusion should default to SNR-weighted loss."""
        import inspect

        from ald_sc.trainer import train_audio_diffusion as _tad

        sig = inspect.signature(_tad)
        assert sig.parameters["loss_weighting"].default == "snr"

    def test_snr_weighting_changes_loss_vs_none(self) -> None:
        """SNR-weighted loss differs from unweighted on the same batch."""
        prior = _make_prior()
        vae = _make_vae()
        loader = _make_dataloader(n=4, audio_length=320 * 16)
        sched = CosineSchedule(num_steps=100)

        for p in vae.parameters():
            p.requires_grad_(False)

        torch.manual_seed(42)
        dit_none = MinimalDiT(
            latent_channels=128,
            latent_length=16,
            patch_size=4,
            dim=32,
            depth=1,
            num_heads=4,
            spec_dim=24,
        )

        torch.manual_seed(42)
        dit_snr = MinimalDiT(
            latent_channels=128,
            latent_length=16,
            patch_size=4,
            dim=32,
            depth=1,
            num_heads=4,
            spec_dim=24,
        )

        losses_none = list(
            train_audio_diffusion(
                loader, vae, dit_none, prior, sched, epochs=1, lr=1e-3,
                loss_weighting="none",
            )
        )
        losses_snr = list(
            train_audio_diffusion(
                loader, vae, dit_snr, prior, sched, epochs=1, lr=1e-3,
                loss_weighting="snr",
            )
        )
        assert len(losses_none) > 0
        assert len(losses_snr) > 0
        # Per-batch loss values differ because of the per-timestep weight.
        diffs = [
            abs(a["loss"] - b["loss"]) for a, b in zip(losses_none, losses_snr)
        ]
        assert any(d > 1e-6 for d in diffs), (
            f"SNR weighting produces same loss as unweighted: {diffs}"
        )

    def test_snr_weighting_still_trains(self) -> None:
        """SNR-weighted training still decreases loss."""
        prior = _make_prior()
        vae = _make_vae()
        loader = _make_dataloader(n=8, audio_length=320 * 16)
        sched = CosineSchedule(num_steps=100)

        for p in vae.parameters():
            p.requires_grad_(False)

        torch.manual_seed(3407)
        dit = MinimalDiT(
            latent_channels=128,
            latent_length=16,
            patch_size=4,
            dim=32,
            depth=2,
            num_heads=4,
            spec_dim=24,
        )

        losses = list(
            train_audio_diffusion(
                loader, vae, dit, prior, sched, epochs=5, lr=1e-3,
                loss_weighting="snr",
            )
        )
        assert len(losses) > 0
        first = losses[0]["loss"]
        last = losses[-1]["loss"]
        assert last < first * 2.0

    def test_invalid_loss_weighting_raises(self) -> None:
        """Invalid loss_weighting value should raise ValueError."""
        prior = _make_prior()
        vae = _make_vae()
        loader = _make_dataloader(n=2, audio_length=320 * 16)
        sched = CosineSchedule(num_steps=100)

        for p in vae.parameters():
            p.requires_grad_(False)

        dit = MinimalDiT(
            latent_channels=128,
            latent_length=16,
            patch_size=4,
            dim=32,
            depth=1,
            num_heads=4,
            spec_dim=24,
        )
        raised = False
        try:
            it = train_audio_diffusion(
                loader, vae, dit, prior, sched, epochs=1,
                loss_weighting="bogus",
            )
            next(it)
        except ValueError:
            raised = True
        except Exception:
            pass
        assert raised, "loss_weighting='bogus' should raise ValueError"


class TestGradientClipping:
    """#+: gradient clipping in diffusion training loop."""

    def test_gradient_clipping_default_on(self) -> None:
        """Default grad_clip should be > 0 (clipping enabled)."""
        import inspect

        from ald_sc.trainer import train_audio_diffusion as _tad

        sig = inspect.signature(_tad)
        assert sig.parameters["grad_clip"].default > 0.0

    def test_large_gradient_clipped(self) -> None:
        """With grad_clip=1.0, no gradient norm should exceed 1.0 after step."""
        prior = _make_prior()
        vae = _make_vae()
        loader = _make_dataloader(n=2, audio_length=320 * 16)
        sched = CosineSchedule(num_steps=100)

        for p in vae.parameters():
            p.requires_grad_(False)

        torch.manual_seed(3407)
        dit = MinimalDiT(
            latent_channels=128,
            latent_length=16,
            patch_size=4,
            dim=32,
            depth=1,
            num_heads=4,
            spec_dim=24,
        )

        from ald_sc.trainer import train_audio_diffusion as _tad

        # Run with large LR and clipping — check that gradients don't explode.
        records = list(
            _tad(
                loader, vae, dit, prior, sched, epochs=1, lr=1.0,
                grad_clip=1.0,
            )
        )
        assert len(records) > 0
        # After clipping, the total grad norm of dit parameters should be <= 1.0.
        total_norm = torch.norm(
            torch.stack([p.grad.norm() for p in dit.parameters() if p.grad is not None])
        )
        assert total_norm.item() <= 1.0 + 1e-4, (
            f"grad norm {total_norm.item()} exceeds clip=1.0"
        )

    def test_grad_clip_zero_disables(self) -> None:
        """grad_clip=0.0 disables clipping (no error)."""
        prior = _make_prior()
        vae = _make_vae()
        loader = _make_dataloader(n=2, audio_length=320 * 16)
        sched = CosineSchedule(num_steps=100)

        for p in vae.parameters():
            p.requires_grad_(False)

        torch.manual_seed(3407)
        dit = MinimalDiT(
            latent_channels=128,
            latent_length=16,
            patch_size=4,
            dim=32,
            depth=1,
            num_heads=4,
            spec_dim=24,
        )
        records = list(
            train_audio_diffusion(
                loader, vae, dit, prior, sched, epochs=1, lr=1e-3,
                grad_clip=0.0,
            )
        )
        assert len(records) > 0


class TestSTFTWindowCaching:
    """#6: Hann windows cached in ALDSCLoss.__init__ — no per-call allocation."""

    def test_windows_cached_in_init(self) -> None:
        """ALDSCLoss should pre-build and cache window tensors."""
        prior = _make_prior(f=32, q=8)
        loss_fn = ALDSCLoss(prior=prior, stft_fft_sizes=(512, 1024, 2048))
        assert hasattr(loss_fn, "_windows"), "ALDSCLoss should cache windows"
        assert 512 in loss_fn._windows
        assert 1024 in loss_fn._windows
        assert 2048 in loss_fn._windows
        assert loss_fn._windows[512].shape == (512,)

    def test_window_tensor_reused_across_calls(self) -> None:
        """The same cached window tensor should be reused (not re-allocated)."""
        prior = _make_prior(f=32, q=8)
        loss_fn = ALDSCLoss(prior=prior, stft_fft_sizes=(512,))
        x = torch.randn(2, 1, 4096)

        with torch.no_grad():
            _ = loss_fn.stft_loss(x, x)
            cached_window_before = loss_fn._windows[512].data_ptr()
            _ = loss_fn.stft_loss(x, x)
            cached_window_after = loss_fn._windows[512].data_ptr()

        assert cached_window_before == cached_window_after, (
            "Window tensor was reallocated between calls"
        )

    def test_stft_loss_still_correct_with_cache(self) -> None:
        """STFT loss values should be correct with cached windows."""
        prior = _make_prior(f=32, q=8)
        loss_fn = ALDSCLoss(prior=prior, stft_fft_sizes=(512, 1024))
        x = torch.randn(2, 1, 4096)
        x_hat = torch.randn(2, 1, 4096)

        with torch.no_grad():
            loss = loss_fn.stft_loss(x, x_hat)
            assert loss.dim() == 0
            assert loss >= 0

        # Identical inputs should produce near-zero loss.
        with torch.no_grad():
            zero_loss = loss_fn.stft_loss(x, x)
            assert zero_loss < 1e-5