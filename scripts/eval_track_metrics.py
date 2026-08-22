"""Arm metrics per PROTOCOL_10S.md: FAD-proxy, diversity, spectral stats.

Compares generated-arm directories (wavs) against a length-matched
reference set built by pairing held-out ESC-50 clips (equal-power join,
same operator as Track-B stitching).

Usage:
    uv run python scripts/eval_track_metrics.py \
        --arm results/track_a --arm results/track_b \
        --esc50 data/esc50/ESC-50-master --out results/track_metrics.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from datetime import datetime
from pathlib import Path

import soundfile
import torch

from ald_sc.data import Esc50Dataset, PairedSegmentDataset
from ald_sc.eval import encodec_pooled_features, fad_score, spectral_centroid, split_files
from ald_sc.eval_stats import (
    bootstrap_fad_ci,
    cross_clip_frame_excess,
    equivalence_verdict,
)


def _frame_clouds(
    waves: list[torch.Tensor],
    encoder,
    device: torch.device,
    k: int = 32,
) -> list[torch.Tensor]:
    """Per-clip (K,F) L2-normalized EnCodec frame clouds (no pooling)."""
    clouds = []
    total = max(w.shape[-1] for w in waves)
    for w in waves:
        x = w.unsqueeze(0).to(device) if w.dim() == 2 else w.to(device)
        with torch.no_grad():
            z = encoder.extract_features(x).squeeze(0)  # (F, T')
        frames = z.T  # (T', F)
        idx = torch.linspace(0, frames.shape[0] - 1, min(k, frames.shape[0])).long()
        sel = frames[idx]
        sel = torch.nn.functional.normalize(sel.float(), dim=-1)
        pad = k - sel.shape[0]
        if pad > 0:
            sel = torch.nn.functional.pad(sel, (0, 0, 0, pad))
        clouds.append(sel)
    del total
    return clouds


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


def load_arm(directory: Path, max_clips: int | None = None) -> list[torch.Tensor]:
    waves = []
    for path in sorted(directory.glob("*.wav")):
        audio, sr = soundfile.read(str(path))
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        waves.append(torch.from_numpy(audio).float().unsqueeze(0))
        if max_clips and len(waves) >= max_clips:
            break
    if not waves:
        raise SystemExit(f"no wavs under {directory}")
    return waves


def pairwise_cosine_distance(feats: torch.Tensor) -> float:
    normed = torch.nn.functional.normalize(feats, dim=-1)
    sims = normed @ normed.T
    n = feats.shape[0]
    off = sims[~torch.eye(n, dtype=torch.bool)]
    return float(1.0 - off.mean().item())


def arm_stats(waves: list[torch.Tensor], sample_rate: int) -> dict[str, float]:
    centroids = torch.cat([spectral_centroid(w, sample_rate) for w in waves])
    rms = torch.stack([w.pow(2).mean().sqrt() for w in waves])
    return {
        "centroid_mean_hz": float(centroids.mean().item()),
        "centroid_std_hz": float(centroids.std(unbiased=False).item()),
        "rms_mean": float(rms.mean().item()),
        "rms_std": float(rms.std(unbiased=False).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PROTOCOL_10S arm metrics")
    parser.add_argument("--arm", action="append", required=True,
                        help="Generated arm directory (repeatable)")
    parser.add_argument("--reference-dir", type=str, default=None,
                        help="Pre-built reference wav directory; otherwise "
                        "built from ESC-50 test split via equal-power pairing")
    parser.add_argument("--esc50", type=str, default="data/esc50/ESC-50-master")
    parser.add_argument("--n-reference", type=int, default=64)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--fad-margin", type=float, default=None,
                        help="Equivalence margin; default derives from the "
                        "--null-dir bootstrap CI (empirical same-generator band)")
    parser.add_argument("--null-dir", type=str, default=None,
                        help="Same-generator arm used to derive the null CI "
                        "(e.g. baselines/v0.12-tracks/ref_bank)")
    parser.add_argument("--out", type=str, default="results/track_metrics.csv")
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device != "auto"
        else torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    )

    from ald_sc.audio_codec import EnCodecEncoder

    encoder = EnCodecEncoder()

    # Reference set: length-matched (paired ~10 s) held-out corpus clips.
    # PROTOCOL forbids zero-padding; pairing mirrors the Track-B operator.
    if args.reference_dir:
        ref_waves = load_arm(Path(args.reference_dir), args.n_reference)
    else:
        root = Path(args.esc50)
        ds = Esc50Dataset(root, audio_length=5 * args.sample_rate)
        files = [str(p) for p in sorted((root / "audio").glob("*.wav"))]
        _, _, test_files = split_files(files, seed=3407)
        index = {str(p): i for i, p in enumerate(ds.files)}
        class _Subset(torch.utils.data.Dataset):
            def __init__(self, paths):
                self.paths = paths

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, i):
                return ds[index[str(self.paths[i])]]

        paired = PairedSegmentDataset(_Subset(test_files), crossfade_samples=480)
        ref_waves = [
            paired[i].cpu() for i in range(min(args.n_reference // 2, len(paired)))
        ]
    print(f"reference clips: {len(ref_waves)}")

    ref_feats = encodec_pooled_features(iter(ref_waves), encoder, device)
    ref_stats = arm_stats(ref_waves, args.sample_rate)

    # Null CI: same-generator variability defines the equivalence margin
    # when none is supplied (protocol: margins from empirical pilot spread).
    margin = args.fad_margin
    null_ci = None
    if args.null_dir:
        null_waves = load_arm(Path(args.null_dir), args.max_clips)
        null_feats = encodec_pooled_features(iter(null_waves), encoder, device)
        null_ci = bootstrap_fad_ci(
            null_feats, ref_feats, n_boot=args.n_boot, alpha=args.alpha
        )
        if margin is None:
            margin = null_ci[1]
        print(f"null CI (same-generator): {null_ci}  -> margin {margin}")

    rows: list[str] = []
    header = (
        "arm,n,fad_vs_ref,fad_ci_low,fad_ci_high,fad_margin,verdict,"
        "frame_excess,pairwise_cos_dist,centroid_mean_hz,centroid_std_hz,"
        "rms_mean,rms_std,duration_s"
    )
    rows.append(header)
    for arm_dir in args.arm:
        path = Path(arm_dir)
        waves = load_arm(path, args.max_clips)
        feats = encodec_pooled_features(iter(waves), encoder, device)
        duration = float(waves[0].shape[-1]) / args.sample_rate
        stats = arm_stats(waves, args.sample_rate)
        ci_low, ci_high = bootstrap_fad_ci(
            feats, ref_feats, n_boot=args.n_boot, alpha=args.alpha
        )
        verdict = (
            equivalence_verdict(ci_low, ci_high, margin)
            if margin is not None
            else "no-margin"
        )
        clouds = _frame_clouds(waves, encoder, device)
        row = {
            "arm": path.name,
            "n": len(waves),
            "fad_vs_ref": round(fad_score(feats, ref_feats), 2),
            "fad_ci_low": round(ci_low, 2),
            "fad_ci_high": round(ci_high, 2),
            "fad_margin": margin,
            "verdict": verdict,
            "frame_excess": round(cross_clip_frame_excess(clouds), 4),
            "pairwise_cos_dist": round(pairwise_cosine_distance(feats), 4),
            **{k: round(v, 4) for k, v in stats.items()},
            "duration_s": round(duration, 2),
        }
        rows.append(",".join(str(v) for v in row.values()))
        print(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    provenance = (
        f"# PROTOCOL_10S arm metrics\n"
        f"# created_at: {datetime.now().isoformat(timespec='seconds')}\n"
        f"# git_commit: {_git_commit()}\n"
        f"# torch: {torch.__version__} device: {device}\n"
        f"# reference_n: {len(ref_waves)} "
        f"(fad_vs_ref ref_stats: {ref_stats})\n"
    )
    out.write_text(provenance + "\n".join(rows) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
