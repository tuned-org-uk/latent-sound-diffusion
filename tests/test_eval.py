"""Tests for the evaluation pipeline (issue #49).

All tests run on CPU with synthetic data so they are fast and hermetic.
They verify the public API of ``ald_sc.eval`` end-to-end.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import torch
from torch import nn

from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.build_prior import build_arrow_prior
from ald_sc.data import ToyAudioDataset, build_audio_dataloader
from ald_sc.eval import (
    ablation_table,
    audio_embedding,
    band_energy_retention,
    clap_proxy_score,
    compression_ratio,
    compression_ratio_vs_n,
    eps_sweep,
    evaluate_reconstruction,
    fad_score,
    frechet_distance,
    midi_pitch_contour,
    reconstruction_table,
    recursive_variant_drift,
    rehydration_coherence,
    spectral_centroid,
    spectral_rolloff,
    split_files,
    text_embedding,
    variant_diversity,
    write_csv,
)
from ald_sc.losses import ALDSCLoss


def _toy_prior(q: int = 4) -> ArrowSpacePrior:
    emb = torch.randn(32, 16)
    return build_arrow_prior(emb, q=q, k=4)


class _StubEncoder(nn.Module):
    """Minimal encoder producing (z, A, c_spec) for reconstruction tests.

    The latent channels (128) are independent of the prior feature dim (16).
    ``encode`` pools z over time to get A of shape (B, latent_channels), then
    projects down to the prior's feature dim F before computing c_spec, so the
    prior's chart_energy_descriptor receives a compatible (B, F) tensor.
    """

    def __init__(
        self,
        latent_channels: int = 128,
        feature_dim: int = 16,
        audio_length: int = 24000,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(1, latent_channels, 320, stride=320)
        self.feat = nn.Linear(latent_channels, feature_dim)
        self.latent_channels = latent_channels
        self.feature_dim = feature_dim
        self.audio_length = audio_length

    def encode(self, x, prior):
        z = self.proj(x).float()
        a_pooled = z.mean(dim=2)
        a = self.feat(a_pooled)
        c_spec = prior.chart_energy_descriptor(a)
        return z, a, c_spec

    def extract_features(self, x):
        return self.proj(x).float()


class _TinyGraphDecoder(nn.Module):
    """Tiny graph-style decoder taking (z, c_spec) for tests."""

    def __init__(self, latent_channels: int = 128, out_channels: int = 1) -> None:
        super().__init__()
        self.proj = nn.Conv1d(latent_channels, out_channels, 1)
        self.up = nn.Upsample(scale_factor=320, mode="nearest")

    def forward(self, z, c_spec=None):
        h = self.proj(self.up(z))
        return h


class _TinyBaselineDecoder(nn.Module):
    """Baseline-style decoder (ignores c_spec). Accepts it optionally so it
    is callable through the graph-decoder code path when not an instance of
    the real ``BaselineAudioDecoder`` (which uses ``isinstance`` dispatch).
    """

    def __init__(self, latent_channels: int = 128, out_channels: int = 1) -> None:
        super().__init__()
        self.proj = nn.Conv1d(latent_channels, out_channels, 1)
        self.up = nn.Upsample(scale_factor=320, mode="nearest")

    def forward(self, z, c_spec=None):
        return self.proj(self.up(z))


class TestSplitFiles:
    def test_three_way_split(self) -> None:
        files = [Path(f"/d/clip_{i:03d}.wav") for i in range(100)]
        train, val, test = split_files(files, 0.7, 0.15, seed=42)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_disjoint(self) -> None:
        files = [Path(f"/d/clip_{i:03d}.wav") for i in range(100)]
        train, val, test = split_files(files, 0.7, 0.15, seed=42)
        all_paths = set(train) | set(val) | set(test)
        assert len(all_paths) == 100

    def test_reproducible(self) -> None:
        files = [Path(f"/d/clip_{i:03d}.wav") for i in range(50)]
        a = split_files(files, seed=7)
        b = split_files(files, seed=7)
        assert a == b

    def test_diff_seeds_differ(self) -> None:
        files = [Path(f"/d/clip_{i:03d}.wav") for i in range(50)]
        a = split_files(files, seed=7)
        b = split_files(files, seed=8)
        assert a != b

    def test_sorted_before_shuffle(self) -> None:
        """Split is deterministic regardless of input order."""
        files_a = [Path(f"/d/clip_{i:03d}.wav") for i in range(20)]
        files_b = list(reversed(files_a))
        assert split_files(files_a, seed=1) == split_files(files_b, seed=1)


class TestReconstruction:
    def test_evaluate_returns_floats(self) -> None:
        prior = _toy_prior()
        encoder = _StubEncoder()
        dec = _TinyGraphDecoder()
        loss_fn = ALDSCLoss(prior=prior, lambda_stft=0.0)
        ds = ToyAudioDataset(num_samples=8, audio_length=24000)
        loader = build_audio_dataloader(ds, batch_size=4, shuffle=False)
        m = evaluate_reconstruction(
            encoder, dec, prior, loader, loss_fn, torch.device("cpu")
        )
        assert isinstance(m, dict)
        for key in ("rec", "stft", "chart", "smooth"):
            assert key in m
            assert isinstance(m[key], float)
            assert m[key] >= 0.0

    def test_ablation_cspec_off_sends_zeros(self) -> None:
        """Without c_spec the decoder receives a zero vector."""
        prior = _toy_prior()
        encoder = _StubEncoder()
        seen: list[torch.Tensor] = []

        class _SpyDecoder(_TinyGraphDecoder):
            def forward(self, z, c_spec=None):
                seen.append(c_spec)
                return super().forward(z, c_spec)

        dec = _SpyDecoder()
        loss_fn = ALDSCLoss(prior=prior, lambda_stft=0.0)
        ds = ToyAudioDataset(num_samples=4, audio_length=24000)
        loader = build_audio_dataloader(ds, batch_size=4, shuffle=False)
        evaluate_reconstruction(
            encoder, dec, prior, loader, loss_fn, torch.device("cpu"), use_cspec=False
        )
        assert all(c is not None and torch.all(c == 0) for c in seen)

    def test_reconstruction_table_rows(self) -> None:
        prior = _toy_prior()
        encoder = _StubEncoder()
        graph = _TinyGraphDecoder()
        baseline = _TinyBaselineDecoder()
        loss_fn = ALDSCLoss(prior=prior, lambda_stft=0.0)
        ds = ToyAudioDataset(num_samples=8, audio_length=24000)
        loader = build_audio_dataloader(ds, batch_size=4, shuffle=False)
        loaders = {"train": loader, "val": loader, "test": loader}
        rows = reconstruction_table(
            encoder, graph, baseline, prior, loss_fn, loaders, torch.device("cpu")
        )
        assert len(rows) == 6
        assert {r["split"] for r in rows} == {"train", "val", "test"}
        assert {r["decoder"] for r in rows} == {"graph", "baseline"}
        for r in rows:
            assert "L1" in r and r["L1"] >= 0

    def test_ablation_table_structure(self) -> None:
        prior = _toy_prior()
        encoder = _StubEncoder()
        dec = _TinyGraphDecoder()
        loss_fn = ALDSCLoss(prior=prior, lambda_stft=0.0)
        ds = ToyAudioDataset(num_samples=4, audio_length=24000)
        loader = build_audio_dataloader(ds, batch_size=4, shuffle=False)
        loaders = {"test": loader}
        rows = ablation_table(
            encoder, dec, prior, loss_fn, loaders, torch.device("cpu")
        )
        assert len(rows) == 3
        labels = [r["with_cspec"] for r in rows]
        assert True in labels and False in labels and "delta" in labels
        delta = [r for r in rows if r["with_cspec"] == "delta"][0]
        with_r = [r for r in rows if r["with_cspec"] is True][0]
        without = [r for r in rows if r["with_cspec"] is False][0]
        assert abs(delta["L1"] - (without["L1"] - with_r["L1"])) < 1e-6


class TestFrechetFAD:
    def test_zero_for_identical(self) -> None:
        a = torch.randn(20, 16)
        assert abs(fad_score(a, a.clone())) < 1e-3

    def test_positive_for_different(self) -> None:
        a = torch.randn(50, 16)
        b = torch.randn(50, 16) + 5.0
        assert fad_score(a, b) > 0

    def test_frechet_shapes(self) -> None:
        a = torch.randn(30, 8)
        b = torch.randn(30, 8)
        assert isinstance(frechet_distance(a, b), float)

    def test_few_samples_fallback(self) -> None:
        a = torch.randn(1, 8)
        b = torch.randn(1, 8) + 10
        val = fad_score(a, b)
        assert val >= 0.0
        assert isinstance(val, float)


class TestEmbeddings:
    def test_text_embedding_normalized(self) -> None:
        v = text_embedding("bass guitar warm", 32)
        assert v.shape == (32,)
        assert abs(v.norm().item() - 1.0) < 1e-5

    def test_text_embedding_deterministic(self) -> None:
        a = text_embedding("same prompt", 16)
        b = text_embedding("same prompt", 16)
        c = text_embedding("other prompt", 16)
        assert torch.allclose(a, b)
        assert not torch.allclose(a, c)

    def test_text_embedding_salt_diverges(self) -> None:
        a = text_embedding("prompt", 16, salt=0)
        b = text_embedding("prompt", 16, salt=1)
        assert not torch.allclose(a, b)

    def test_audio_embedding_single_clip(self) -> None:
        prior = _toy_prior()
        enc = _StubEncoder()
        clip = torch.randn(1, 1, 24000)
        v = audio_embedding(clip, enc, prior, torch.device("cpu"))
        assert v.shape[0] == 128
        assert abs(v.norm().item() - 1.0) < 1e-5

    def test_clap_proxy_score_in_range(self) -> None:
        enc = _StubEncoder()
        clips = [torch.randn(1, 1, 24000) for _ in range(4)]
        score = clap_proxy_score(clips, "warm bass", enc, torch.device("cpu"))
        assert -1.0 <= score <= 1.0


class TestCompression:
    def test_ratio_grows_with_n(self) -> None:
        prior = _toy_prior()
        r = compression_ratio_vs_n([1, 10, 100, 1000], 96000, prior)
        ratios = [row["compression_ratio"] for row in r]
        for i in range(1, len(ratios)):
            assert ratios[i] > ratios[i - 1], "ratio should increase with N"

    def test_dehydrated_has_prior_once(self) -> None:
        prior = _toy_prior()
        r1 = compression_ratio(1, 96000, prior)
        r10 = compression_ratio(10, 96000, prior)
        assert abs(r1["prior_bits"] - r10["prior_bits"]) < 1e-9

    def test_raw_vs_dehydrated(self) -> None:
        prior = _toy_prior()
        r = compression_ratio(100, 96000, prior)
        assert r["raw_bits"] > r["dehydrated_bits"]
        assert r["compression_ratio"] > 1.0

    def test_code_rate_mode(self) -> None:
        prior = _toy_prior()
        r = compression_ratio(
            10, 48000, prior, use_code_rate=True, encod_bandwidth_kbps=24
        )
        expected_latent = 24 * 1000 * (48000 / 24000)
        assert abs(r["latent_bits_per_clip"] - expected_latent) < 1e-3


class TestSpectralCentroid:
    def test_sine_at_known_freq(self) -> None:
        sr = 24000
        t = torch.arange(sr, dtype=torch.float32) / sr
        freq = 1000.0
        x = torch.sin(2 * torch.pi * freq * t)
        c = spectral_centroid(x, sample_rate=sr)
        assert c.dim() == 1
        assert c.shape[0] > 0
        assert c.mean().item() > 0

    def test_shape_variants(self) -> None:
        x = torch.randn(24000)
        for shape_fn in [
            lambda t: t,
            lambda t: t.unsqueeze(0),
            lambda t: t.unsqueeze(0).unsqueeze(0),
        ]:
            c = spectral_centroid(shape_fn(x))
            assert c.dim() == 1


class TestMidiContour:
    def test_contour_active_frames(self) -> None:
        events = [(60, 0.0, 0.5), (64, 0.5, 0.5)]
        num_frames = 100
        contour, mask = midi_pitch_contour(
            events, num_frames, sample_rate=24000, hop=240
        )
        assert contour.shape[0] == num_frames
        assert mask.shape[0] == num_frames
        assert mask.sum() > 0
        assert (contour[mask] > 0).all()


class TestRehydrationCoherence:
    def test_returns_dict(self) -> None:
        class _FakeModel:
            sample_rate = 24000

            def synthesize_midi(self, events, bank, pitch_bank_root, seed):
                t = (
                    torch.arange(self.sample_rate, dtype=torch.float32)
                    / self.sample_rate
                )
                return torch.sin(2 * torch.pi * 220 * t).unsqueeze(0)

        events = [(60, 0.0, 0.25), (64, 0.25, 0.25), (67, 0.5, 0.25)]
        bank = [torch.randn(1, 24000)]
        result = rehydration_coherence(_FakeModel(), events, bank, pitch_bank_root=60)
        assert "pearson_r" in result
        assert "midi_events" in result and result["midi_events"] == 3
        assert "active_frames" in result


class TestVariantDiversity:
    def test_depth_one_zero_distance(self) -> None:
        enc = _StubEncoder()

        def make(d):
            return [torch.randn(1, 1, 24000) for _ in range(d)]

        rows = variant_diversity(make, [1, 4, 8], enc, torch.device("cpu"))
        assert rows[0]["depth"] == 1
        assert rows[0]["mean_distance"] == 0.0
        assert rows[1]["mean_distance"] >= 0.0
        assert rows[2]["n_variants"] == 8

    def test_distances_nonneg(self) -> None:
        enc = _StubEncoder()

        def make(d):
            return [torch.randn(1, 1, 24000) for _ in range(d)]

        rows = variant_diversity(make, [2, 4], enc, torch.device("cpu"))
        for r in rows:
            assert r["mean_distance"] >= 0.0
            assert r["min_distance"] >= 0.0


class TestBandEnergyRetention:
    def _loader(self):
        ds = ToyAudioDataset(num_samples=8, audio_length=24000)
        return build_audio_dataloader(ds, batch_size=4, shuffle=False)

    def test_one_row_per_mode(self) -> None:
        prior = _toy_prior(q=4)
        encoder = _StubEncoder()
        dec = _TinyGraphDecoder()
        rows = band_energy_retention(
            encoder, dec, prior, self._loader(), torch.device("cpu")
        )
        assert len(rows) == prior.q
        for r in rows:
            assert {"k", "e_orig", "e_recon", "retention", "cosine"} <= set(r)
            assert r["e_orig"] >= 0.0 and r["e_recon"] >= 0.0
            assert -1.0 <= r["cosine"] <= 1.0

    def test_normalized_band_energies(self) -> None:
        prior = _toy_prior(q=4)
        encoder = _StubEncoder()
        dec = _TinyGraphDecoder()
        rows = band_energy_retention(
            encoder, dec, prior, self._loader(), torch.device("cpu")
        )
        total = sum(r["e_orig"] for r in rows)
        assert abs(total - 1.0) < 0.05  # e_tilde sums to ~1

    def test_cspec_off_sends_zeros(self) -> None:
        prior = _toy_prior(q=4)
        encoder = _StubEncoder()
        captured: dict[str, object] = {}

        class _SpyDecoder(_TinyGraphDecoder):
            def forward(self, z, c_spec=None):  # type: ignore[override]
                captured["c_spec"] = c_spec
                return super().forward(z, c_spec)

        band_energy_retention(
            encoder,
            _SpyDecoder(),
            prior,
            self._loader(),
            torch.device("cpu"),
            use_cspec=False,
        )
        assert captured["c_spec"] is not None
        assert float(captured["c_spec"].abs().sum()) == 0.0  # type: ignore[union-attr]

    def test_retention_finite(self) -> None:
        prior = _toy_prior(q=4)
        encoder = _StubEncoder()
        dec = _TinyGraphDecoder()
        rows = band_energy_retention(
            encoder, dec, prior, self._loader(), torch.device("cpu")
        )
        for r in rows:
            assert r["retention"] == r["retention"]  # not NaN
            assert r["retention"] >= 0.0


class TestSpectralRolloff:
    def test_sine_rolloff_below_high_freq(self) -> None:
        sr = 24000
        t = torch.arange(sr, dtype=torch.float32) / sr
        x = torch.sin(2 * torch.pi * 500.0 * t)
        ro = spectral_rolloff(x, sample_rate=sr)
        assert ro.dim() == 1
        assert 0.0 < ro.mean().item() < 5000.0

    def test_noise_rolloff_above_sine(self) -> None:
        sr = 24000
        t = torch.arange(sr, dtype=torch.float32) / sr
        sine = torch.sin(2 * torch.pi * 500.0 * t)
        noise = torch.randn(sr) * 0.1
        ro_sine = spectral_rolloff(sine, sample_rate=sr).mean().item()
        ro_noise = spectral_rolloff(noise, sample_rate=sr).mean().item()
        assert ro_noise > ro_sine

    def test_shape_variants(self) -> None:
        x = torch.randn(24000)
        for shape_fn in [
            lambda t: t,
            lambda t: t.unsqueeze(0),
            lambda t: t.unsqueeze(0).unsqueeze(0),
        ]:
            ro = spectral_rolloff(shape_fn(x))
            assert ro.dim() == 1
            assert (ro >= 0.0).all()
            assert (ro <= 12000.0 + 1.0).all()  # <= Nyquist


class TestRecursiveVariantDrift:
    def test_rows_per_round(self) -> None:
        class _FakeModel:
            sample_rate = 24000

            def generate_sound_bank(self, n, steps, seed):
                return [torch.randn(1, 24000) for _ in range(n)]

            def synthesize_midi(self, events, bank, seed=None, **kw):
                t = (
                    torch.arange(self.sample_rate, dtype=torch.float32)
                    / self.sample_rate
                )
                return torch.sin(2 * torch.pi * (220 + 10 * seed) * t).unsqueeze(0)

            def condition_on_audio(self, audio, n, steps, strength, seed):
                return [torch.randn(1, 24000) for _ in range(n)]

        enc = _StubEncoder()
        events = [(60, 0.0, 0.25), (64, 0.25, 0.25)]
        rows = recursive_variant_drift(
            _FakeModel(),
            events,
            rounds=2,
            encoder=enc,
            device=torch.device("cpu"),
            steps=2,
            seed=3407,
            bank_n=2,
        )
        assert len(rows) == 3  # round 0 + 2 recursive rounds
        for r in rows:
            assert {
                "round",
                "centroid_mean_hz",
                "centroid_std_hz",
                "rolloff_mean_hz",
                "clap_distance_to_round0",
            } <= set(r)
            assert r["centroid_mean_hz"] > 0.0
            assert r["rolloff_mean_hz"] > 0.0
            assert r["clap_distance_to_round0"] >= 0.0
        assert rows[0]["round"] == 0
        # Round 0's CLAP-proxy distance to itself is zero up to float32
        # round-off (1 - cos_sim of recomputed identical features); exact
        # equality flakes across platforms (issue #62 CI run, ~6e-8).
        assert rows[0]["clap_distance_to_round0"] < 1e-6
        assert [r["round"] for r in rows] == [0, 1, 2]

    def test_true_recursion_feeds_output_back(self) -> None:
        calls: dict[str, list[object]] = {"condition": [], "synthesize": 0}

        class _FakeModel:
            sample_rate = 24000

            def generate_sound_bank(self, n, steps, seed):
                return [torch.randn(1, 24000) for _ in range(n)]

            def synthesize_midi(self, events, bank, seed=None, **kw):
                calls["synthesize"] = calls["synthesize"] + 1
                t = (
                    torch.arange(self.sample_rate, dtype=torch.float32)
                    / self.sample_rate
                )
                return torch.sin(2 * torch.pi * 300 * t).unsqueeze(0)

            def condition_on_audio(self, audio, n, steps, strength, seed):
                calls["condition"].append(audio)
                return [torch.randn(1, 24000) for _ in range(n)]

        enc = _StubEncoder()
        events = [(60, 0.0, 0.25)]
        recursive_variant_drift(
            _FakeModel(),
            events,
            rounds=3,
            encoder=enc,
            device=torch.device("cpu"),
            steps=2,
            seed=3407,
            bank_n=2,
        )
        # condition_on_audio called once per recursive round...
        assert len(calls["condition"]) == 3
        # ...and each call receives the previous round's RENDER (finite audio)
        for a in calls["condition"]:
            assert isinstance(a, torch.Tensor)
            assert float(a.abs().sum()) > 0.0
        # synthesize called once per round + once for round 0
        assert calls["synthesize"] == 4


class TestEpsSweep:
    def test_rows_and_bounds(self) -> None:
        from ald_sc.dit import MinimalDiT
        from ald_sc.schedule import CosineSchedule

        prior = build_arrow_prior(torch.randn(32, 128), q=4, k=4)
        dit = MinimalDiT(
            latent_channels=128,
            latent_length=16,
            patch_size=4,
            dim=32,
            depth=1,
            num_heads=2,
            text_dim=0,
            spec_dim=12,
        )
        sched = CosineSchedule(num_steps=100)
        dec = _TinyGraphDecoder()
        enc = _StubEncoder()
        ref = torch.randn(4, 128)
        rows = eps_sweep(
            dit,
            sched,
            prior,
            dec,
            enc,
            ref,
            [1e-4, 1e-3, 1e-2],
            torch.device("cpu"),
            n_gen=1,
            steps=4,
        )
        assert len(rows) == 3
        assert [r["eps"] for r in rows] == [1e-4, 1e-3, 1e-2]
        for r in rows:
            assert 0.0 <= r["mean_steps"] <= 4.0
            assert r["FAD"] >= 0.0
            assert r["FAD_method"] == "encod_proxy"

    def test_larger_eps_stops_earlier(self) -> None:
        from ald_sc.dit import MinimalDiT
        from ald_sc.schedule import CosineSchedule

        prior = build_arrow_prior(torch.randn(32, 128), q=4, k=4)
        dit = MinimalDiT(
            latent_channels=128,
            latent_length=16,
            patch_size=4,
            dim=32,
            depth=1,
            num_heads=2,
            text_dim=0,
            spec_dim=12,
        )
        sched = CosineSchedule(num_steps=100)
        dec = _TinyGraphDecoder()
        enc = _StubEncoder()
        ref = torch.randn(4, 128)
        rows = eps_sweep(
            dit,
            sched,
            prior,
            dec,
            enc,
            ref,
            [1e-6, 1.0],
            torch.device("cpu"),
            n_gen=1,
            steps=8,
        )
        assert rows[0]["mean_steps"] >= rows[1]["mean_steps"]


class TestWriteCSV:
    def test_writes_header_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "out.csv"
            write_csv(path, [{"a": 1, "b": 2.0}, {"a": 3, "b": 4.0}])
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["a"] == "1"
            assert rows[1]["b"] == "4.0"

    def test_empty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            write_csv(path, [])
            assert path.exists()
            assert path.read_text() == ""
