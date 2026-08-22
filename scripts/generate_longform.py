"""Generate Track-A long-form samples from a v0.12 DiT (issue #60).

Loads the checkpoint produced by scripts/train_audio_longform.py together
with its *_metadata.json provenance, validates geometry, and renders an
eval-ready bank per PROTOCOL_10S.md (default seed block 901000+).

Usage:
    uv run python scripts/generate_longform.py \
        --dit dit_v0.12_10s.pt --prior results/artifacts/esc50_prior.pt \
        --graph-dec results/artifacts/esc50_graph_dec.pt \
        --n 16 --out-dir results/track_a
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
from ald_sc.config import validate_dit_state_dict
from ald_sc.dit import MinimalDiT
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.inference import BANK_MODES, LSDModel
from ald_sc.schedule import CosineSchedule


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
    parser = argparse.ArgumentParser(description="Track-A long-form sampling")
    parser.add_argument("--dit", type=str, required=True)
    parser.add_argument("--prior", type=str, required=True)
    parser.add_argument("--graph-dec", type=str, required=True)
    parser.add_argument("--metadata", type=str, default=None,
                        help="Defaults to <dit stem>_metadata.json; 'none' "
                        "reads geometry from the --geo-* flags instead")
    parser.add_argument("--geo-latent-length", type=int, default=None)
    parser.add_argument("--geo-latent-channels", type=int, default=128)
    parser.add_argument("--geo-patch-size", type=int, default=8)
    parser.add_argument("--geo-dim", type=int, default=64)
    parser.add_argument("--geo-depth", type=int, default=2)
    parser.add_argument("--geo-num-heads", type=int, default=4)
    parser.add_argument("--geo-spec-dim", type=int, default=24)
    parser.add_argument("--base-channels", type=int, default=32,
                        help="GraphDecoder width used at training time")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed-start", type=int, default=901000,
                        help="PROTOCOL_10S Track-A eval block")
    parser.add_argument("--bank-mode", type=str, default="canonical",
                        choices=list(BANK_MODES))
    parser.add_argument("--num-steps-schedule", type=int, default=1000)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="results/track_a")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = torch.device(args.device)

    dit_path = Path(args.dit)
    if args.metadata == "none":
        if args.geo_latent_length is None:
            raise SystemExit("--metadata none requires --geo-latent-length")
        geo = {
            "latent_length": args.geo_latent_length,
            "latent_channels": args.geo_latent_channels,
            "patch_size": args.geo_patch_size,
            "dim": args.geo_dim,
            "depth": args.geo_depth,
            "num_heads": args.geo_num_heads,
            "spec_dim": args.geo_spec_dim,
        }
        meta = {"git_commit": None}
    else:
        meta_path = (
            Path(args.metadata)
            if args.metadata
            else dit_path.with_name(dit_path.stem + "_metadata.json")
        )
        if not meta_path.exists():
            raise SystemExit(f"metadata not found: {meta_path} (pass --metadata)")
        meta = json.loads(meta_path.read_text())
        geo = meta["geometry"]

    dit = MinimalDiT(
        latent_channels=int(geo["latent_channels"]),
        latent_length=int(geo["latent_length"]),
        patch_size=int(geo["patch_size"]),
        dim=int(geo["dim"]),
        depth=int(geo["depth"]),
        num_heads=int(geo["num_heads"]),
        spec_dim=int(geo["spec_dim"]),
    )
    state_dict = torch.load(dit_path, weights_only=False)
    validate_dit_state_dict(
        state_dict,
        latent_channels=int(geo["latent_channels"]),
        latent_length=int(geo["latent_length"]),
        patch_size=int(geo["patch_size"]),
    )
    dit.load_state_dict(state_dict)
    dit = dit.to(device).eval()

    prior = torch.load(args.prior, weights_only=False)
    prior = prior.to(device)

    graph_dec = GraphDecoder(
        latent_channels=int(geo["latent_channels"]),
        out_channels=1,
        feature_dim=int(geo["latent_channels"]),
        base_channels=args.base_channels,
        prior=prior,
        upsample_strides=(2, 4, 5, 8),
    )
    graph_dec.load_state_dict(torch.load(args.graph_dec, weights_only=False))
    graph_dec = graph_dec.to(device).eval()

    encoder = EnCodecEncoder(sample_rate=args.sample_rate, bandwidth=24)
    schedule = CosineSchedule(num_steps=args.num_steps_schedule)

    model = LSDModel(
        prior=prior,
        dit=dit,
        decoder=graph_dec,
        encoder=encoder,
        schedule=schedule,
        sample_rate=args.sample_rate,
    )

    print(
        f"generating n={args.n} @ {geo['latent_length']} frames "
        f"({geo['latent_length'] * (1 / 75):.1f}s), steps={args.steps}, "
        f"mode={args.bank_mode}, seeds {args.seed_start}..{args.seed_start + args.n - 1} "
        f"on {device}"
    )
    bank = model.generate_sound_bank(
        n=args.n,
        steps=args.steps,
        temperature=args.temperature,
        seed=args.seed_start,
        bank_mode=args.bank_mode,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, clip in enumerate(bank):
        fname = f"{i:03d}.wav"
        soundfile.write(
            str(out_dir / fname), clip.squeeze(0).cpu().numpy(), args.sample_rate
        )
        clips.append({"file": fname, "seed": args.seed_start + i})

    manifest = {
        "arm": "track_a",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "device": str(device),
        "checkpoint": str(dit_path),
        "checkpoint_metadata": args.metadata == "none" and "none" or str(meta_path),
        "training_git_commit": meta.get("git_commit"),
        "resolved_seeds": [args.seed_start + i for i in range(len(bank))],
        "steps": args.steps,
        "temperature": args.temperature,
        "bank_mode": args.bank_mode,
        "latent_frames": int(geo["latent_length"]),
        "clips": clips,
        "protocol": "PROTOCOL_10S.md",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(bank)} clips + manifest.json to {out_dir}")


if __name__ == "__main__":
    main()
