"""Materialise a generated bank through MIDI patterns (Mode C audition).

Loads rendered bank clips (.wav) and plays them as pitched notes over
built-in musical patterns, so stem quality can be judged in context
without LSD-studio.

Usage:
    uv run python scripts/materialise_midi.py \
        --bank baselines/v0.12-tracks/confirm_a \
        --out-dir baselines/v0.12-tracks/midi_renders
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import soundfile
import torch

from ald_sc.audio_codec import EnCodecEncoder
from ald_sc.build_prior import load_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.inference import LSDModel
from ald_sc.schedule import CosineSchedule

MAJOR = [0, 2, 4, 5, 7, 9, 11]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"


def pattern_arpeggio(
    root: int, bars: int, bpm: float
) -> list[tuple[int, float, float]]:
    """Eighth-note up-down arpeggio over the major triad + octave."""
    step = 60.0 / bpm / 2.0
    seq = [0, 4, 7, 12, 7, 4]
    return [
        (root + seq[i % len(seq)] + 12 * ((i // len(seq)) % 2), i * step, step * 0.95)
        for i in range(int(bars * 4 * (60.0 / bpm) / step))
    ]


def pattern_chords(root: int, bars: int, bpm: float) -> list[tuple[int, float, float]]:
    """Whole-bar triads: I–vi–IV–V."""
    bar = 60.0 / bpm * 4
    degrees = [0, 9, 5, 7]
    events = []
    for bar_i, deg in enumerate(degrees[:bars]):
        base = root + deg - 12
        for ivl in (0, 4, 7):
            events.append((base + ivl, bar_i * bar, bar * 0.98))
    return events


def pattern_scale(root: int, bars: int, bpm: float) -> list[tuple[int, float, float]]:
    """Quarter-note ascending major scale, two octaves."""
    step = 60.0 / bpm
    notes = [root + MAJOR[i % 7] + 12 * (i // 7) for i in range(14)]
    return [(n, i * step, step * 0.9) for i, n in enumerate(notes[: int(bars * 4)])]


def pattern_melody(root: int, bars: int, bpm: float) -> list[tuple[int, float, float]]:
    """Fixed motif with mixed durations (quarter/eighth/dotted)."""
    beat = 60.0 / bpm
    motif = [(0, 1.0), (4, 0.5), (7, 0.5), (9, 1.0), (7, 0.75), (4, 0.25)]
    events, t = [], 0.0
    total = bars * 4 * beat
    i = 0
    while t < total:
        deg, beats = motif[i % len(motif)]
        events.append((root + deg, t, beats * beat * 0.92))
        t += beats * beat
        i += 1
    return events


PATTERNS = {
    "arpeggio": pattern_arpeggio,
    "chords": pattern_chords,
    "scale": pattern_scale,
    "melody": pattern_melody,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="MIDI materialise a bank")
    parser.add_argument(
        "--bank", type=str, required=True, help="Directory of bank .wav clips"
    )
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--dit",
        type=str,
        default=None,
        help="Checkpoint backing the bank (provenance only)",
    )
    parser.add_argument("--prior", type=str, default="results/artifacts/esc50_prior.pt")
    parser.add_argument(
        "--graph-dec", type=str, default="results/artifacts/esc50_graph_dec.pt"
    )
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument(
        "--patterns",
        type=str,
        default="all",
        help="Comma list from: " + ", ".join(PATTERNS),
    )
    parser.add_argument("--root", type=int, default=60)
    parser.add_argument("--bars", type=int, default=4)
    parser.add_argument("--bpm", type=float, default=100.0)
    parser.add_argument("--max-clips", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--barontini",
        action="store_true",
        help="Regenerate the bank with the Barontini "
        "heat-death clock (SpectralSchedule) instead of "
        "loading --bank wavs; requires --dit",
    )
    parser.add_argument(
        "--clock-eps",
        type=float,
        default=1e-2,
        help="Normalized remaining-dissipation threshold",
    )
    parser.add_argument("--clock-horizon", type=float, default=1.0)
    parser.add_argument(
        "--seed-start",
        type=int,
        default=901200,
        help="Seed block for regenerated banks",
    )
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device != "auto"
        else torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    )

    names = sorted(PATTERNS) if args.patterns == "all" else args.patterns.split(",")

    # Bank: load the already-generated clips — no diffusion sampling here.
    bank_dir = Path(args.bank)
    paths = sorted(bank_dir.glob("*.wav"))[: max(1, args.max_clips)]
    if not paths:
        raise SystemExit(f"no wavs under {bank_dir}")
    bank = []
    for p in paths:
        audio, sr = soundfile.read(str(p))
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if sr != args.sample_rate:
            raise SystemExit(f"{p.name}: sample rate {sr} != {args.sample_rate}")
        bank.append(torch.from_numpy(audio).float().unsqueeze(0))

    # Model bundle: needed for sample_rate contract + future mode-B flows;
    # materialisation itself is DSP-only over the loaded bank.
    geo_path = Path(
        args.metadata
        or (Path(args.dit).with_name(Path(args.dit).stem + "_metadata.json"))
        if args.dit
        else "baselines/v0.12-tracks/dit_v0.12_10s_metadata.json"
    )
    geo = (
        json.loads(geo_path.read_text())["geometry"]
        if geo_path.exists()
        else {"latent_channels": 128}
    )
    prior = load_arrow_prior(args.prior).to(device)
    graph_dec = GraphDecoder(
        latent_channels=int(geo.get("latent_channels", 128)),
        out_channels=1,
        feature_dim=int(geo.get("latent_channels", 128)),
        base_channels=32,
        prior=prior,
        upsample_strides=(2, 4, 5, 8),
    )
    state_dict = (
        torch.load(args.graph_dec, weights_only=True, map_location="cpu")
        if args.graph_dec
        else None
    )
    if state_dict:
        graph_dec.load_state_dict(state_dict)
    graph_dec = graph_dec.to(device).eval()

    if args.barontini:
        if not args.dit:
            raise SystemExit("--barontini requires --dit (bank is regenerated)")
        meta_path = Path(args.dit).with_name(Path(args.dit).stem + "_metadata.json")
        geo_dit = (
            json.loads(meta_path.read_text())["geometry"]
            if meta_path.exists()
            else {
                "latent_channels": 128,
                "latent_length": 375,
                "patch_size": 8,
                "dim": 256,
                "depth": 4,
                "num_heads": 4,
                "spec_dim": 24,
            }
        )
        dit = MinimalDiT(
            latent_channels=int(geo_dit["latent_channels"]),
            latent_length=int(geo_dit["latent_length"]),
            patch_size=int(geo_dit["patch_size"]),
            dim=int(geo_dit["dim"]),
            depth=int(geo_dit["depth"]),
            num_heads=int(geo_dit["num_heads"]),
            spec_dim=int(geo_dit["spec_dim"]),
        )
        from ald_sc.config import validate_dit_state_dict

        state_dict = torch.load(args.dit, weights_only=True, map_location="cpu")
        validate_dit_state_dict(
            state_dict,
            latent_channels=int(geo_dit["latent_channels"]),
            latent_length=int(geo_dit["latent_length"]),
            patch_size=int(geo_dit["patch_size"]),
        )
        dit.load_state_dict(state_dict)
        dit = dit.to(device).eval()
    else:
        dit = MinimalDiT(latent_channels=128)  # unused: bank comes from wavs

    model = LSDModel(
        prior=prior,
        dit=dit,
        decoder=graph_dec,
        encoder=EnCodecEncoder(),
        schedule=CosineSchedule(num_steps=1000),
        sample_rate=args.sample_rate,
    )

    if args.barontini:
        from ald_sc.spectral_schedule import SpectralSchedule

        model.spectral_schedule = SpectralSchedule(
            prior, horizon=args.clock_horizon, eps=args.clock_eps
        )
        print(
            f"Barontini clock engaged: eps={args.clock_eps}, "
            f"horizon={args.clock_horizon}; regenerating bank from {args.dit}"
        )
        bank = model.generate_sound_bank(
            n=max(1, args.max_clips),
            steps=50,
            seed=args.seed_start,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in names:
        events = PATTERNS[name.strip()](args.root, args.bars, args.bpm)
        render = model.synthesize_midi(
            events, bank, pitch_bank_root=args.root, seed=3407
        )
        fname = f"materialised_{name.strip()}.wav"
        soundfile.write(
            str(out_dir / fname),
            render.squeeze(0).cpu().numpy(),
            args.sample_rate,
        )
        dur = render.shape[-1] / args.sample_rate
        entries.append(
            {
                "file": fname,
                "pattern": name.strip(),
                "duration_s": round(dur, 2),
                "events": len(events),
            }
        )
        print(f"{fname}: {len(events)} events, {dur:.2f}s")

    manifest = {
        "arm": "midi_materialised",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "device": str(device),
        "source_bank": str(bank_dir),
        "checkpoint": args.dit,
        "patterns": names,
        "root_midi": args.root,
        "bars": args.bars,
        "bpm": args.bpm,
        "renders": entries,
        "protocol": "PROTOCOL_10S.md",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(entries)} renders + manifest.json to {out_dir}")


if __name__ == "__main__":
    main()
