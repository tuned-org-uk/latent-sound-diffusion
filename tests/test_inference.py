"""Tests for the LSD inference contract (sound-bank / condition / MIDI)."""

from __future__ import annotations


import json
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.inference import Bank, LSDModel
from ald_sc.schedule import CosineSchedule


class StubEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, stride: int = 320) -> None:
        super().__init__()
        self.proj = nn.Conv1d(1, latent_dim, stride, stride=stride)

    def encode(self, x: Tensor, prior) -> tuple[Tensor, Tensor, Tensor]:
        z = self.proj(x).float()
        a = z.mean(dim=2)
        return z, a, prior.chart_energy_descriptor(a)

    def extract_features(self, x: Tensor) -> Tensor:
        return self.proj(x).float()


def _make_model(latent_length: int = 16) -> LSDModel:
    torch.manual_seed(3407)
    embeddings = torch.randn(32, 128)
    prior = build_arrow_prior(embeddings, q=8, k=4)
    encoder = StubEncoder()
    decoder = GraphDecoder(128, 1, 128, 16, prior, (2, 4, 5, 8))
    dit = MinimalDiT(
        latent_channels=128,
        latent_length=latent_length,
        patch_size=4,
        dim=32,
        depth=1,
        num_heads=4,
        spec_dim=24,
    )
    sched = CosineSchedule(num_steps=100)
    return LSDModel(
        prior=prior,
        dit=dit,
        decoder=decoder,
        encoder=encoder,
        schedule=sched,
        sample_rate=24000,
    )


class TestGenerateSoundBank:
    def test_returns_n_clips(self) -> None:
        m = _make_model()
        bank = m.generate_sound_bank(n=4, steps=4, seed=3407)
        assert len(bank) == 4
        for clip in bank:
            assert clip.shape[0] == 1
            assert clip.shape[1] == 16 * 320  # latent_length * stride

    def test_seed_reproducible(self) -> None:
        m = _make_model()
        a = m.generate_sound_bank(n=2, steps=3, seed=42)
        b = m.generate_sound_bank(n=2, steps=3, seed=42)
        for ca, cb in zip(a, b):
            assert torch.allclose(ca, cb)

    def test_seed_none_runs(self) -> None:
        m = _make_model()
        bank = m.generate_sound_bank(n=2, steps=3, seed=None)
        assert len(bank) == 2

    def test_temperature_changes_output(self) -> None:
        m = _make_model()
        low = m.generate_sound_bank(n=1, steps=3, seed=1, temperature=0.1)[0]
        high = m.generate_sound_bank(n=1, steps=3, seed=1, temperature=2.0)[0]
        assert not torch.allclose(low, high)


class TestBankModes:
    """Issue #53: latent-variation bank modes around the canonical draw."""

    def test_all_modes_return_n_clips(self) -> None:
        from ald_sc.inference import BANK_MODES

        m = _make_model()
        for mode in BANK_MODES:
            bank = m.generate_sound_bank(n=3, steps=4, seed=3407, bank_mode=mode)
            assert len(bank) == 3, mode
            for clip in bank:
                assert clip.shape == (1, 16 * 320), mode

    def test_default_mode_is_canonical(self) -> None:
        m = _make_model()
        default = m.generate_sound_bank(n=2, steps=3, seed=11)
        canonical = m.generate_sound_bank(n=2, steps=3, seed=11, bank_mode="canonical")
        for a, b in zip(default, canonical):
            assert torch.allclose(a, b)

    def test_jitter_diversifies(self) -> None:
        m = _make_model()
        bank = m.generate_sound_bank(
            n=4, steps=4, seed=3407, bank_mode="jitter", bank_variety=0.5
        )
        l1 = [
            float((bank[i] - bank[j]).abs().mean())
            for i in range(4)
            for j in range(i + 1, 4)
        ]
        assert sum(l1) / len(l1) > 0.01

    def test_jitter_first_clip_is_canonical(self) -> None:
        """variety=0 must reduce every mode to the canonical draw."""
        m = _make_model()
        canon = m.generate_sound_bank(n=2, steps=3, seed=5, bank_mode="canonical")[0]
        for mode in ("jitter", "residual"):
            bank = m.generate_sound_bank(
                n=2, steps=3, seed=5, bank_mode=mode, bank_variety=0.0
            )
            assert torch.allclose(bank[0], canon, atol=1e-5), mode
            assert torch.allclose(bank[1], canon, atol=1e-5), mode

    def test_stopvar_grid_semantics(self) -> None:
        """stopvar = same seed, step counts spread over the trajectory.

        With steps=4, n=3, bank_variety=0.5 the grid is lo=round(4*0.5)=2
        to hi=round(4*0.98)=4, i.e. {2, 3, 4}: each latent must equal a
        fresh DDIM draw at that step count with the SAME seed.
        """
        from ald_sc.sampling import sample_ddim

        m = _make_model()
        latents = m._bank_latents(
            n=3, steps=4, seed=5, bank_mode="stopvar", bank_variety=0.5
        )
        for z, s in zip(latents, (2, 3, 4)):
            ref = sample_ddim(
                m.dit, m.schedule, batch_size=1, steps=s, seed=5, device=z.device
            )
            assert torch.allclose(z, ref, atol=1e-6)

    def test_stopvar_floor_follows_variety(self) -> None:
        """bank_variety is the stop-time floor: lowest grid entry follows it."""
        from ald_sc.sampling import sample_ddim

        m = _make_model()
        latents = m._bank_latents(
            n=2, steps=10, seed=5, bank_mode="stopvar", bank_variety=0.24
        )
        # floor round(10 * 0.24) = 2; ceil entry round(10 * 0.98) = 10
        ref_lo = sample_ddim(m.dit, m.schedule, batch_size=1, steps=2, seed=5)
        ref_hi = sample_ddim(m.dit, m.schedule, batch_size=1, steps=10, seed=5)
        assert torch.allclose(latents[0], ref_lo.to(latents[0].device), atol=1e-6)
        assert torch.allclose(latents[1], ref_hi.to(latents[1].device), atol=1e-6)

    def test_generated_banks_are_dc_free(self) -> None:
        """DC-block: outputs must carry no DC offset (PR #59 feedback fix).

        The raw decoder output has a large DC component (+0.4..+0.56
        measured); _normalize must remove it before peak-normalising.
        """
        m = _make_model()
        for mode in ("canonical", "jitter", "residual", "stopvar"):
            bank = m.generate_sound_bank(n=2, steps=3, seed=5, bank_mode=mode)
            for clip in bank:
                assert abs(float(clip.mean())) < 1e-5, mode

    def test_modes_reproducible(self) -> None:
        m = _make_model()
        for mode in ("jitter", "residual", "stopvar"):
            a = m.generate_sound_bank(n=3, steps=4, seed=99, bank_mode=mode)
            b = m.generate_sound_bank(n=3, steps=4, seed=99, bank_mode=mode)
            for ca, cb in zip(a, b):
                assert torch.allclose(ca, cb), mode

    def test_invalid_mode_raises(self) -> None:
        m = _make_model()
        with pytest.raises(ValueError, match="bank_mode"):
            m.generate_sound_bank(n=1, steps=2, seed=1, bank_mode="bogus")

    def test_negative_variety_raises(self) -> None:
        m = _make_model()
        with pytest.raises(ValueError, match="bank_variety"):
            m.generate_sound_bank(n=1, steps=2, seed=1, bank_variety=-0.1)

    def test_from_generation_passes_mode(self) -> None:
        m = _make_model()
        bank = Bank.from_generation(
            m, n=3, steps=3, seed=7, bank_mode="jitter", bank_variety=0.25
        )
        assert len(bank.clips) == 3

    def test_n_one_supported(self) -> None:
        m = _make_model()
        for mode in ("residual", "stopvar"):
            bank = m.generate_sound_bank(n=1, steps=3, seed=3, bank_mode=mode)
            assert len(bank) == 1


class TestConditionOnAudio:
    def test_variants_shape(self) -> None:
        m = _make_model()
        audio = torch.randn(1, 1, 16 * 320)
        variants = m.condition_on_audio(audio, n=3, steps=3, strength=0.5, seed=3407)
        assert len(variants) == 3
        for v in variants:
            assert v.shape[0] == 1
            assert v.shape[1] == 16 * 320

    def test_seed_reproducible(self) -> None:
        m = _make_model()
        audio = torch.randn(1, 1, 16 * 320)
        a = m.condition_on_audio(audio, n=2, steps=2, strength=0.3, seed=7)
        b = m.condition_on_audio(audio, n=2, steps=2, strength=0.3, seed=7)
        for va, vb in zip(a, b):
            assert torch.allclose(va, vb)

    def test_strength_changes_output(self) -> None:
        m = _make_model()
        audio = torch.randn(1, 1, 16 * 320)
        weak = m.condition_on_audio(audio, n=1, steps=2, strength=0.1, seed=5)[0]
        strong = m.condition_on_audio(audio, n=1, steps=2, strength=1.0, seed=5)[0]
        assert not torch.allclose(weak, strong)


class TestSynthesizeMidi:
    def test_requires_non_empty_bank(self) -> None:
        m = _make_model()
        try:
            import pytest
        except ImportError:
            return
        with pytest.raises(ValueError, match="bank"):
            m.synthesize_midi([(60, 0.0, 0.5)], bank=[], seed=3407)

    def test_midi_to_audio_length_proportional(self) -> None:
        m = _make_model()
        bank = m.generate_sound_bank(n=1, steps=3, seed=3407)
        short = m.synthesize_midi([(60, 0.0, 0.5)], bank=bank, seed=3407)
        long = m.synthesize_midi([(60, 0.0, 1.0)], bank=bank, seed=3407)
        assert short.shape[-1] < long.shape[-1]

    def test_pitched_bank_sound_is_retimed(self) -> None:
        m = _make_model()
        bank = m.generate_sound_bank(n=1, steps=3, seed=3407)
        out = m.synthesize_midi(
            [(69, 0.0, 0.25), (72, 0.25, 0.25)],
            bank=bank,
            pitch_bank_root=69,
            seed=3407,
        )
        assert out.shape[-1] > bank[0].shape[-1] // 2


class TestStoreModel:
    def _model(self) -> LSDModel:
        return _make_model()

    def test_store_creates_dir_with_artifacts(self, tmp_path) -> None:
        m = self._model()
        out = m.store(root_dir=tmp_path, slug="test")
        out = Path(out)
        assert out.exists() and out.is_dir()
        for name in ("prior.pt", "decoder.pt", "dit.pt", "metadata.json"):
            assert (out / name).exists(), f"missing {name}"
        # slug should appear in dir name
        assert "test" in out.name
        # timestamp prefix YYYYMMDD-HHMMSS
        ts = out.name.split("-test")[0]
        assert len(ts) == 15 and ts[8] == "-"

    def test_store_default_slug_used(self, tmp_path) -> None:
        m = self._model()
        out = Path(m.store(root_dir=tmp_path))
        assert "lsd-model" in out.name

    def test_metadata_json_reproducible_fields(self, tmp_path) -> None:
        m = self._model()
        out = Path(
            m.store(
                root_dir=tmp_path,
                slug="m",
                hyperparams=dict(noise_std=0.1, seed=7, q=8),
            )
        )
        meta = json.loads((out / "metadata.json").read_text())
        assert meta["hyperparameters"]["noise_std"] == 0.1
        assert meta["hyperparameters"]["seed"] == 7
        assert meta["hyperparameters"]["q"] == 8
        assert meta["sample_rate"] == 24000
        assert "decoder_type" in meta
        assert "created_at" in meta

    def test_store_metadata_records_decoder_type(self, tmp_path) -> None:
        m = self._model()  # GraphDecoder
        out = Path(m.store(root_dir=tmp_path, slug="g"))
        meta = json.loads((out / "metadata.json").read_text())
        assert meta["decoder_type"].endswith("GraphDecoder")


class TestBankStore:
    def test_store_writes_wavs_and_manifest(self, tmp_path) -> None:
        m = _make_model(latent_length=16)
        bank = Bank(
            model=m, clips=m.generate_sound_bank(n=3, steps=2, seed=3407), name="tight"
        )
        out = Path(tmp_path) / "modeldir"
        out.mkdir()
        bank_dir = bank.store(out_dir=out)
        bank_dir = Path(bank_dir)
        assert bank_dir.exists() and bank_dir.is_dir()
        wavs = sorted(bank_dir.glob("*.wav"))
        assert len(wavs) == 3
        assert (bank_dir / "manifest.json").exists()
        meta = json.loads((bank_dir / "manifest.json").read_text())
        assert meta["name"] == "tight"
        assert meta["n_clips"] == 3
        assert meta["sample_rate"] == 24000
        assert len(meta["clips"]) == 3
        # clips named 00.wav, 01.wav, ...
        assert wavs[0].name == "00.wav"

    def test_empty_clips_rejected(self) -> None:
        m = _make_model()
        with pytest.raises(ValueError, match="clips"):
            Bank(model=m, clips=[], name="empty")


class TestBankNameSanitization:
    """Bank.store must not let `name` escape out_dir/banks (issue #60)."""

    def test_traversal_name_is_sanitized(self, tmp_path) -> None:
        m = _make_model()
        bank = Bank(model=m, clips=[torch.zeros(1, 100)], name="../../evil")
        out = bank.store(tmp_path)
        assert tmp_path in out.parents or out.parent == tmp_path / "banks"
        assert "evil" in str(out)
        assert ".." not in str(out.relative_to(tmp_path))
        assert (out / "00.wav").exists()

    def test_safe_name_unchanged(self, tmp_path) -> None:
        m = _make_model()
        bank = Bank(model=m, clips=[torch.zeros(1, 100)], name="my-bank_1")
        out = bank.store(tmp_path)
        assert out == tmp_path / "banks" / "my-bank_1"


class TestSynthesizeMidiCorrectness:
    """Regression tests for the Mode-C render bugs (issue #60 deferred bundle).

    v0.11 re-timed the transposed clip by resampling a second time to the
    requested duration, which cancelled the pitch shift entirely: rendered
    pitch depended only on src_len/duration, not on the MIDI note.
    """

    def _tone_model(self, freq_hz: float = 220.0) -> tuple[LSDModel, torch.Tensor]:
        m = _make_model()
        sr = m.sample_rate
        t = torch.arange(int(1.0 * sr)) / sr
        tone = torch.sin(2 * torch.pi * freq_hz * t).unsqueeze(0)
        return m, tone.unsqueeze(0)

    def _dominant_freq(self, wave: torch.Tensor, sr: int) -> float:
        spec = torch.abs(torch.fft.rfft(wave.float()))
        return float(spec.argmax().item()) * sr / wave.shape[-1]

    def test_transposition_survives_duration_fit(self) -> None:
        m, tone = self._tone_model(220.0)
        out = m.synthesize_midi(
            [(72, 0.0, 0.25)], bank=[tone], pitch_bank_root=60, seed=3407
        )
        seg = out[0, : int(0.2 * m.sample_rate)]
        f = self._dominant_freq(seg, m.sample_rate)
        assert 380.0 < f < 500.0, (
            f"+12 semitones from 220 Hz must render near 440 Hz; got {f:.1f}"
        )

    def test_untransposed_note_keeps_source_pitch(self) -> None:
        m, tone = self._tone_model(220.0)
        out = m.synthesize_midi(
            [(60, 0.0, 0.5)], bank=[tone], pitch_bank_root=60, seed=3407
        )
        seg = out[0, : int(0.3 * m.sample_rate)]
        f = self._dominant_freq(seg, m.sample_rate)
        assert 200.0 < f < 245.0, f"root note must stay ~220 Hz; got {f:.1f}"

    def test_invalid_events_are_skipped_with_warning(self) -> None:
        import structlog

        m, tone = self._tone_model()
        with structlog.testing.capture_logs() as caps:
            out = m.synthesize_midi(
                [
                    (60, -0.5, 0.5),  # negative start: wraparound risk
                    (60, float("nan"), 0.5),  # non-finite start
                    (60, 0.0, -1.0),  # negative duration
                    (60, 0.0, 0.1),  # the one valid event
                ],
                bank=[tone],
                seed=3407,
            )
        warnings = [e for e in caps if e.get("event") == "midi_event_invalid"]
        assert len(warnings) == 3, warnings
        assert out.shape[-1] >= int(0.1 * m.sample_rate)

    def test_all_events_invalid_raises(self) -> None:
        m, tone = self._tone_model()
        import pytest

        with pytest.raises(ValueError, match="no valid"):
            m.synthesize_midi([(60, -1.0, 0.5)], bank=[tone], seed=3407)

    def test_note_edges_are_faded_against_clicks(self) -> None:
        """Constant-level clip would hard-step at onset without a fade."""
        m = _make_model()
        flat = torch.ones(1, m.sample_rate // 2)
        out = m.synthesize_midi([(60, 0.0, 0.4)], bank=[flat], seed=3407)
        # A click is a sample-to-sample step; the raised-cosine fade bounds
        # the onset slope far below the full-scale step (~1.0).
        onset_slope = out[0, :72].diff().abs().max().item()
        assert onset_slope < 0.1, (
            f"onset must ramp smoothly; max step {onset_slope:.3f}"
        )

    def test_bank_selection_follows_composer_order_not_start_time(self) -> None:
        m = _make_model()
        sr = m.sample_rate
        t = torch.arange(sr // 2) / sr
        low = torch.sin(2 * torch.pi * 220.0 * t).unsqueeze(0)
        high = torch.sin(2 * torch.pi * 880.0 * t).unsqueeze(0)
        # Composer lists the LOW sound first even though it starts LATER;
        # timbre must follow list position, not chronological sorting.
        out = m.synthesize_midi(
            [(60, 1.0, 0.4), (60, 0.0, 0.4)],
            bank=[low, high],
            pitch_bank_root=60,
            seed=3407,
        )
        early = self._dominant_freq(out[0, : int(0.3 * sr)], sr)
        late = self._dominant_freq(out[0, int(1.05 * sr) : int(1.3 * sr)], sr)
        assert early > 600, (
            f"second-listed sound plays first chronologically; f={early:.0f}"
        )
        assert late < 400, f"first-listed sound must take the later slot; f={late:.0f}"


class TestArtefactHardening:
    """Safe loads + digest manifests (issue #60 deferred bundle)."""

    def test_load_arrow_prior_accepts_state_dict_format(self, tmp_path) -> None:
        from ald_sc.build_prior import load_arrow_prior

        m = _make_model()
        path = tmp_path / "prior.pt"
        torch.save(m.prior.state_dict(), path)

        prior = load_arrow_prior(path)

        for key in ("L_F", "U_q", "eigvals_q", "lambdas_ed"):
            assert torch.allclose(getattr(prior, key), getattr(m.prior, key)), key

    def test_load_arrow_prior_legacy_pickle_falls_back_with_warning(
        self, tmp_path
    ) -> None:
        import structlog

        from ald_sc.build_prior import load_arrow_prior

        m = _make_model()
        path = tmp_path / "legacy_prior.pt"
        torch.save(m.prior, path)  # whole-object pickle: v0.11 format

        with structlog.testing.capture_logs() as caps:
            prior = load_arrow_prior(path)
        assert any(e.get("event") == "legacy_pickle_artifact" for e in caps)
        assert torch.allclose(prior.L_F, m.prior.L_F)

    def test_store_writes_state_dict_prior_and_digest_manifest(self, tmp_path) -> None:
        import hashlib

        out = _make_model().store(root_dir=tmp_path, slug="hardened")
        # Prior must now be a safe state-dict artefact.
        sd = torch.load(out / "prior.pt", weights_only=True)
        assert {"L_F", "U_q", "eigvals_q", "lambdas_ed"} <= set(sd)

        manifest = out / "MANIFEST.sha256"
        assert manifest.exists()
        for line in manifest.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
            assert digest == actual, name
        assert {line.split()[1] for line in manifest.read_text().splitlines()} == {
            "prior.pt",
            "decoder.pt",
            "dit.pt",
            "metadata.json",
        }

    def test_bank_store_writes_digest_manifest(self, tmp_path) -> None:
        import hashlib

        m = _make_model()
        bank = Bank(model=m, clips=[torch.zeros(1, 100)], name="digested")
        out = bank.store(tmp_path)
        manifest = out / "MANIFEST.sha256"
        assert manifest.exists()
        for line in manifest.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
            assert digest == actual, name
