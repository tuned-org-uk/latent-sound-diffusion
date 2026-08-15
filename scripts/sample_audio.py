"""Generate audio by sampling from the trained DiT and decoding.

Usage:
    uv run python scripts/sample_audio.py --prior prior.pt --decoder decoder.pt --dit dit.pt --out results/sample.wav
    uv run python scripts/sample_audio.py --toy --out results/sample.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import soundfile

from ald_sc.audio_codec import BaselineAudioDecoder
from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.schedule import CosineSchedule
from ald_sc.sampling import sample_ddim


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio via ALD-SC")
    parser.add_argument("--prior", type=str, default=None)
    parser.add_argument("--decoder", type=str, default=None)
    parser.add_argument("--dit", type=str, default=None)
    parser.add_argument("--graph", action="store_true", help="Use graph decoder")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--latent-length", type=int, default=75)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--out", type=str, default="results/sample.wav")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Build or load prior
    if args.prior and Path(args.prior).exists():
        prior = torch.load(args.prior, weights_only=False)
    else:
        embeddings = torch.randn(64, 128)
        prior = build_arrow_prior(embeddings, q=args.q, k=4)
    prior = prior.to(device)

    # Build DiT
    dit = MinimalDiT(
        latent_channels=args.latent_channels,
        latent_length=args.latent_length,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        spec_dim=3 * args.q,
    )
    if args.dit and Path(args.dit).exists():
        dit.load_state_dict(torch.load(args.dit, weights_only=False))
    dit = dit.to(device).eval()
    sched = CosineSchedule(num_steps=args.num_steps)

    # Sample latent
    print(f"Sampling latent (steps={args.steps}, seed={args.seed})...")
    z = sample_ddim(
        dit, sched, batch_size=1, steps=args.steps, seed=args.seed, device=device
    )
    print(f"Sampled z shape: {z.shape}")

    # Derive c_spec from z (self-consistent decoding)
    a = z.mean(dim=2)
    c_spec = prior.chart_energy_descriptor(a)
    print(f"c_spec shape: {c_spec.shape}")

    # Decode
    if args.graph:
        decoder = GraphDecoder(
            latent_channels=args.latent_channels,
            out_channels=1,
            feature_dim=128,
            base_channels=64,
            prior=prior,
        )
    else:
        decoder = BaselineAudioDecoder(
            latent_channels=args.latent_channels,
            out_channels=1,
            base_channels=64,
        )

    if args.decoder and Path(args.decoder).exists():
        decoder.load_state_dict(torch.load(args.decoder, weights_only=False))
    decoder = decoder.to(device).eval()

    with torch.no_grad():
        if isinstance(decoder, BaselineAudioDecoder):
            audio = decoder(z)
        else:
            audio = decoder(z, c_spec)

    print(f"Audio shape: {audio.shape}")

    # Normalize and save
    audio = audio.squeeze(0)  # (1, T) -> (T,) or keep (1, T)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # (1, T)
    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(out_path), audio.squeeze(0).numpy(), args.sample_rate)
    print(f"Saved audio to {out_path} ({audio.shape[1] / args.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
