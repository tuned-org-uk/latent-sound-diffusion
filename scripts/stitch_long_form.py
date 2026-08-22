"""Stitch bank clips into one long-form render (Track B control arm).

Waveform-domain equal-power overlap-add over existing checkpoints — no
retraining required. Usage:
    uv run python scripts/stitch_long_form.py --bank results/banks/bank \
        --overlap-ms 20 --out results/long_form.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile
import torch

from ald_sc.stitching import equal_power_overlap_add


def main() -> None:
    parser = argparse.ArgumentParser(description="Track-B long-form stitching")
    parser.add_argument(
        "--bank",
        type=str,
        required=True,
        help="Directory of NN.wav clips (Bank.store layout)",
    )
    parser.add_argument("--out", type=str, default="results/long_form.wav")
    parser.add_argument("--overlap-ms", type=float, default=20.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument(
        "--rms-match",
        action="store_true",
        help="RMS-match clip bodies before stitching",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Seed for clip-order shuffling (0 = keep order)",
    )
    args = parser.parse_args()

    clips = sorted(Path(args.bank).glob("*.wav"))
    if len(clips) < 2:
        raise SystemExit(
            f"--bank must contain >= 2 .wav files; found {len(clips)} in {args.bank}"
        )

    waves = []
    for path in clips:
        audio, sr = soundfile.read(str(path))
        if sr != args.sample_rate:
            raise SystemExit(
                f"{path.name}: sample rate {sr} != expected {args.sample_rate}"
            )
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)  # downmix stereo to mono
        waves.append(torch.from_numpy(audio).float().unsqueeze(0))

    gen = torch.Generator().manual_seed(args.seed)
    if args.seed > 0:
        perm = torch.randperm(len(waves), generator=gen).tolist()
        waves = [waves[i] for i in perm]
        print(f"clip order shuffled with seed {args.seed}")

    if args.rms_match:
        matched = []
        for wave in waves:
            body = wave[:, wave.shape[-1] // 4 :]
            rms = body.pow(2).mean().sqrt()
            scale = 1.0 / rms if rms > 1e-8 else 1.0
            matched.append(wave * scale)
        waves = matched
        print(f"RMS-matched {len(waves)} clips")

    overlap = max(1, int(args.overlap_ms / 1000 * args.sample_rate))
    render = equal_power_overlap_add(waves, overlap=overlap)
    peak = render.abs().max()
    if peak > 1.0:
        render = render / peak

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(out_path), render.squeeze(0).numpy(), args.sample_rate)
    duration = render.shape[-1] / args.sample_rate
    print(
        f"stitched {len(clips)} clips, overlap={args.overlap_ms}ms "
        f"-> {out_path} ({duration:.2f}s)"
    )


if __name__ == "__main__":
    main()
