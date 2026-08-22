"""Track-A long-form DiT training (issue #60): variable-length latent crops.

Trains MinimalDiT across a range of latent durations (default 300-750
EnCodec frames = 4-10 s) using pos-embed interpolation, on segments built
from short archive clips via equal-power pairing when needed.

Usage:
    uv run python scripts/train_audio_longform.py \
        --data-root sound-archive/ --pair --prior prior.pt \
        --min-latent 300 --max-latent 750 --epochs 50 --device auto

Provenance (PROTOCOL_10S.md): resolved seed, git commit, torch version,
device and full geometry are written next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch

from ald_sc.build_prior import build_arrow_prior, load_arrow_prior
from ald_sc.config import load_config, resolve_geometry
from ald_sc.data import (
    AudioFolderDataset,
    Esc50Dataset,
    MusicSynthDataset,
    PairedSegmentDataset,
    build_audio_dataloader,
)
from ald_sc.dit import MinimalDiT
from ald_sc.schedule import CosineSchedule
from ald_sc.trainer import log_training, train_audio_diffusion


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 1-D DiT over variable-length latents (long-form)"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config; CLI flags override")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=str, default=None)
    source.add_argument("--esc50", type=str, default=None)
    source.add_argument("--toy", action="store_true")
    parser.add_argument("--clip-sec", type=float, default=5.0,
                        help="Base clip length in seconds (dataset level)")
    parser.add_argument("--pair", action="store_true",
                        help="Pair consecutive clips into 2x segments "
                        "(equal-power crossfade) for long-form supply")
    parser.add_argument("--crossfade-ms", type=float, default=20.0)
    parser.add_argument("--min-latent", type=int, default=None,
                        help="Min latent frames per batch crop (default 300)")
    parser.add_argument("--max-latent", type=int, default=None,
                        help="Max latent frames; also sizes the pos table "
                        "(default 750, clamped to available audio)")
    parser.add_argument("--prior", type=str, default=None)
    parser.add_argument("--latent-channels", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto | cpu | cuda | mps")
    parser.add_argument("--out", type=str, default="dit_long.pt")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)

    raw_cfg = load_config(args.config)
    geometry = resolve_geometry(
        raw_cfg,
        overrides={
            "latent_channels": args.latent_channels,
            "patch_size": args.patch_size,
            "dim": args.dim,
            "depth": args.depth,
            "num_heads": args.num_heads,
        },
    )

    sample_rate = 24000
    clip_samples = int(args.clip_sec * sample_rate)

    # Prior (required outside --toy; no silent random fallbacks)
    if args.prior and Path(args.prior).exists():
        prior = load_arrow_prior(args.prior)
    elif args.toy:
        gen = torch.Generator().manual_seed(args.seed)
        embeddings = torch.randn(
            64, int(geometry["latent_channels"]), generator=gen
        )
        print("WARNING: seeded random fallback prior (--toy only)")
        prior = build_arrow_prior(embeddings, q=args.q, k=4)
    else:
        raise SystemExit(
            "--prior is required (seeded random prior only under --toy)"
        )
    prior = prior.to(device)

    # Dataset
    if args.toy:
        base = MusicSynthDataset(num_samples=16, audio_length=clip_samples, seed=args.seed)
    elif args.data_root:
        base = AudioFolderDataset(args.data_root, audio_length=clip_samples)
    else:
        base = Esc50Dataset(args.esc50, audio_length=clip_samples)

    crossfade = int(args.crossfade_ms / 1000 * sample_rate)
    dataset = PairedSegmentDataset(base, crossfade_samples=crossfade) if args.pair else base
    if len(dataset) == 0:
        raise SystemExit("dataset resolved to 0 segments; check --data-root/--esc50/--pair")
    segment_samples = (
        2 * clip_samples - min(crossfade, clip_samples - 1) if args.pair else clip_samples
    )

    # Crop window: default 300..750 clamped by what the segments can supply
    available_frames = segment_samples // 320
    max_latent = args.max_latent or 750
    min_latent = args.min_latent or 300
    max_latent = max(1, min(max_latent, available_frames))
    min_latent = max(1, min(min_latent, max_latent))
    print(
        f"segments={len(dataset)} ({segment_samples / sample_rate:.2f}s each) "
        f"crop range {min_latent}..{max_latent} latent frames"
    )

    # Encoder (frozen); stub keeps --toy free of EnCodec downloads
    if args.toy:
        from torch import nn

        class StubEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Conv1d(1, geometry["latent_channels"], 320, stride=320)

            def encode(self, x, prior):
                z = self.proj(x).float()
                a = z.mean(dim=2)
                c_spec = prior.chart_energy_descriptor(a)
                return z, a, c_spec

            def extract_features(self, x):
                return self.proj(x).float()

        encoder = StubEncoder()
    else:
        from ald_sc.audio_codec import EnCodecEncoder

        encoder = EnCodecEncoder()

    class _EncoderOnly(torch.nn.Module):
        """Trainer contract needs only `.encoder`; no decoder is trained here."""

        def __init__(self, enc):
            super().__init__()
            self.encoder = enc

    vae = _EncoderOnly(encoder)

    dit = MinimalDiT(
        latent_channels=geometry["latent_channels"],
        latent_length=max_latent,
        patch_size=geometry["patch_size"],
        dim=geometry["dim"],
        depth=geometry["depth"],
        num_heads=geometry["num_heads"],
        spec_dim=3 * args.q,
    )
    sched = CosineSchedule(num_steps=args.num_steps)

    loader = build_audio_dataloader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"training on {device} for {args.epochs} epochs ...")

    records = list(
        log_training(
            train_audio_diffusion(
                loader,
                vae,
                dit,
                prior,
                sched,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                latent_crop_range=(min_latent, max_latent),
            ),
            label="Long-form DiT",
        )
    )
    if not records:
        raise SystemExit("no training steps ran; dataset/loader produced no batches")

    losses = [r["loss"] for r in records]
    lengths_seen = sorted({r["latent_len"] for r in records})
    print(f"initial loss {losses[0]:.4f} -> final loss {losses[-1]:.4f}")
    print(f"latent lengths exercised: {lengths_seen}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dit.state_dict(), out_path)
    from safetensors.torch import save_file

    safe_path = str(out_path.with_suffix(".safetensors"))
    save_file(dit.state_dict(), safe_path)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/train_audio_longform.py",
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "device": str(device),
        "resolved_seed": args.seed,
        "geometry": {
            **{k: int(v) for k, v in geometry.items()},
            "latent_length": max_latent,
            "num_patches": -(-max_latent // geometry["patch_size"]),
            "spec_dim": 3 * args.q,
        },
        "latent_crop_range": [min_latent, max_latent],
        "paired_segments": bool(args.pair),
        "clip_seconds": args.clip_sec,
        "crossfade_ms": args.crossfade_ms,
        "num_steps_schedule": args.num_steps,
        "loss_initial": float(losses[0]),
        "loss_final": float(losses[-1]),
        "protocol": "PROTOCOL_10S.md",
    }
    meta_path = out_path.with_name(out_path.stem + "_metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"saved {out_path} (+ .safetensors) and {meta_path}")


if __name__ == "__main__":
    main()
