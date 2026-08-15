"""Tests for the 1-D Mallat scattering transform (issue #48)."""

from __future__ import annotations

import numpy as np
import torch

from ald_sc.scattering1d import (
    MorletFilterBank,
    Scattering1D,
    ScatteringConfig,
    scattering_pooled_features,
)


def _chirp(n: int = 8000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / n
    return np.sin(2 * np.pi * (200 * t + 800 * t * t)) + 0.1 * rng.standard_normal(n)


class TestScatteringConfig:
    def test_default_t_is_two_pow_j(self) -> None:
        cfg = ScatteringConfig(J=5)
        assert cfg.T == 32

    def test_rejects_bad_order(self) -> None:
        try:
            ScatteringConfig(order=3)
        except ValueError:
            pass
        else:
            raise AssertionError("order=3 should raise ValueError")


class TestMorletFilterBank:
    def test_dc_corrected(self) -> None:
        fb = MorletFilterBank(1024, ScatteringConfig())
        for _xi, _sigma, psi_hat in fb.wavelets:
            assert abs(psi_hat[0]) < 1e-12

    def test_wavelets_unit_energy(self) -> None:
        fb = MorletFilterBank(1024, ScatteringConfig())
        for _xi, _sigma, psi_hat in fb.wavelets:
            assert abs((np.abs(psi_hat) ** 2).sum() - 1.0) < 1e-6

    def test_low_pass_unit_sum(self) -> None:
        fb = MorletFilterBank(1024, ScatteringConfig())
        assert abs(fb.phi_hat.sum() - 1.0) < 1e-6


class TestScattering1D:
    def test_feature_dim_formula(self) -> None:
        cfg = ScatteringConfig(J=5, Q=4, order=2)
        sc = Scattering1D(1024, cfg)
        # 1 (order 0) + J*Q (order 1) + C(J,2)*Q^2 (order 2, j2 > j1)
        assert sc.feature_dim == 1 + 20 + 10 * 16
        assert sc.transform(np.random.default_rng(0).standard_normal(1024)).shape == (
            sc.feature_dim,
        )

    def test_order1_dim(self) -> None:
        cfg = ScatteringConfig(J=4, Q=2, order=1)
        sc = Scattering1D(512, cfg)
        assert sc.feature_dim == 1 + 8

    def test_deterministic(self) -> None:
        sc = Scattering1D(1024, ScatteringConfig())
        x = _chirp(1024)
        assert torch.equal(
            torch.from_numpy(sc.transform(x)), torch.from_numpy(sc.transform(x))
        )

    def test_wrong_length_raises(self) -> None:
        sc = Scattering1D(1024)
        try:
            sc.transform(np.zeros(999))
        except ValueError:
            pass
        else:
            raise AssertionError("length mismatch should raise ValueError")

    def test_translation_invariance(self) -> None:
        """40-sample shift changes features < 1% (issue #48 property)."""
        n, shift = 8000, 40
        sc = Scattering1D(n, ScatteringConfig())
        x = _chirp(n, seed=7)
        f0 = sc.transform(x)
        f1 = sc.transform(np.roll(x, shift))
        rel = float(np.abs(f1 - f0).sum() / (np.abs(f0).sum() + 1e-12))
        assert rel < 0.01, f"relative change {rel:.4f} exceeds 1%"

    def test_shift_compression_vs_raw(self) -> None:
        """Feature-space sensitivity is far below raw-signal sensitivity."""
        n, shift = 8000, 40
        sc = Scattering1D(n, ScatteringConfig())
        x = _chirp(n, seed=7)
        raw = float(np.abs(np.roll(x, shift) - x).sum() / (np.abs(x).sum() + 1e-12))
        f0, f1 = sc.transform(x), sc.transform(np.roll(x, shift))
        feat = float(np.abs(f1 - f0).sum() / (np.abs(f0).sum() + 1e-12))
        assert feat < raw / 10


class TestScatteringPooledFeatures:
    def test_batches_clips_to_n_by_d(self) -> None:
        clips = [torch.randn(1, 1, 4000) for _ in range(3)]
        feats = scattering_pooled_features(clips, signal_length=4000)
        assert feats.shape == (3, Scattering1D(4000).feature_dim)
        assert feats.dtype == torch.float32

    def test_truncates_and_pads(self) -> None:
        long_clip = torch.randn(1, 1, 6000)
        short_clip = torch.randn(1, 1, 2000)
        feats = scattering_pooled_features([long_clip, short_clip], signal_length=4000)
        assert feats.shape[0] == 2
        assert not torch.isnan(feats).any()

    def test_accepts_loader_batches(self) -> None:
        batches = [torch.randn(2, 1, 4000), torch.randn(1, 4000), torch.randn(3, 4000)]
        feats = scattering_pooled_features(batches, signal_length=4000)
        assert feats.shape[0] == 6

    def test_multichannel_batch_meaned(self) -> None:
        batch = torch.randn(2, 3, 4000)  # (B, C, T)
        feats = scattering_pooled_features([batch], signal_length=4000)
        assert feats.shape[0] == 2

    def test_no_hang_on_batch_tensor(self) -> None:
        """Regression: (B, 1, T) batch with B > 1 must not loop forever."""
        import time

        t0 = time.time()
        feats = scattering_pooled_features([torch.randn(2, 1, 4000)], signal_length=4000)
        assert feats.shape[0] == 2
        assert time.time() - t0 < 30
