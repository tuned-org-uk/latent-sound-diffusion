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
    parser.add_argument("--audio-length", type=int, default=24000)
    parser.add_argument(
        "--latent-length",
        type=int,
        default=None,
        help="Latent length (frames). If not given, derived as "
        "audio_length // 320 (EnCodec stride).",
    )
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Diversity knob mapped to DDIM eta (clamped 0-1). "
        "0 = deterministic, 1 = full stochastic (DDPM-equivalent).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale. 1.0 = pure conditional, "
        "0.0 = unconditional, >1.0 = amplified conditioning.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate. Outputs are saved as "
        "sample_00.wav, sample_01.wav, etc. (or sample.wav if n=1).",
    )
    parser.add_argument(
        "--per-sample-conditioning",
        action="store_true",
        default=False,
        help="Resample c_spec independently for each generated sample. "
        "Produces a more varied sound bank at the cost of spectral "
        "coherence across samples. Default: shared c_spec (coherent bank).",
    )
    parser.add_argument("--out", type=str, default="results/sample.wav")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Derive latent_length from audio_length if not explicitly set (#28 #4).
    latent_length = args.latent_length
    if latent_length is None:
        latent_length = args.audio_length // 320

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
        latent_length=latent_length,
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

    from ald_sc.inference import _temperature_to_eta

    eta = _temperature_to_eta(args.temperature)

    # Compute shared c_spec probe (used unless --per-sample-conditioning).
    with torch.no_grad():
        z_probe = torch.randn(1, args.latent_channels, latent_length, device=device)
        a_probe = z_probe.mean(dim=2)
        c_spec_shared = prior.chart_energy_descriptor(a_probe)

    # Decode setup
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating {args.num_samples} sample(s) "
        f"(steps={args.steps}, seed={args.seed}, eta={eta}, "
        f"guidance_scale={args.guidance_scale}, "
        f"per_sample_conditioning={args.per_sample_conditioning})..."
    )

    for idx in range(args.num_samples):
        sample_seed = args.seed + idx

        if args.per_sample_conditioning:
            g = torch.Generator(device=device).manual_seed(sample_seed + 1000)
            z_probe_i = torch.randn(
                1, args.latent_channels, latent_length, device=device, generator=g
            )
            c_spec_i = prior.chart_energy_descriptor(z_probe_i.mean(dim=2))
        else:
            c_spec_i = c_spec_shared

        with torch.no_grad():
            z = sample_ddim(
                dit,
                sched,
                c_spec=c_spec_i,
                batch_size=1,
                steps=args.steps,
                seed=sample_seed,
                device=device,
                eta=eta,
                guidance_scale=args.guidance_scale,
            )

            # Derive c_spec from z for self-consistent decoding
            a = z.mean(dim=2)
            c_spec = prior.chart_energy_descriptor(a)

            if isinstance(decoder, BaselineAudioDecoder):
                audio = decoder(z)
            else:
                audio = decoder(z, c_spec)

        # Normalize
        audio = audio.squeeze(0)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak

        if args.num_samples == 1:
            fname = out_path
        else:
            fname = out_path.parent / f"{out_path.stem}_{idx:02d}{out_path.suffix}"

        soundfile.write(str(fname), audio.squeeze(0).numpy(), args.sample_rate)
        print(
            f"  [{idx + 1}/{args.num_samples}] Saved to {fname} ({audio.shape[1] / args.sample_rate:.2f}s)"
        )

    print("Done.")


if __name__ == "__main__":
    main()
