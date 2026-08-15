"""Train the 1-D DiT denoiser on EnCodec latents.

Usage:
    uv run python scripts/train_audio_diffusion.py --prior prior.pt --decoder decoder.pt --epochs 50
    uv run python scripts/train_audio_diffusion.py --toy --epochs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ald_sc.audio_codec import AudioVAE, BaselineAudioDecoder, EnCodecEncoder
from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.data import ToyAudioDataset, build_audio_dataloader
from ald_sc.schedule import CosineSchedule
from ald_sc.trainer import train_audio_diffusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 1-D DiT on audio latents")
    parser.add_argument("--prior", type=str, default=None)
    parser.add_argument("--decoder", type=str, default=None)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--audio-length", type=int, default=24000)
    parser.add_argument("--latent-channels", type=int, default=128)
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--out", type=str, default="dit.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Build dataset
    if args.toy:
        dataset = ToyAudioDataset(num_samples=32, audio_length=args.audio_length)
    else:
        print("Please use --toy for now (ESC-50 support requires download)")
        return
    loader = build_audio_dataloader(dataset, batch_size=args.batch_size, shuffle=True)

    # Build encoder + decoder (both frozen)
    if args.toy:
        from torch import nn

        class StubEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Conv1d(1, 128, 320, stride=320)

            def encode(self, x, prior):
                z = self.proj(x).float()
                a = z.mean(dim=2)
                c_spec = prior.chart_energy_descriptor(a)
                return z, a, c_spec

            def extract_features(self, x):
                return self.proj(x).float()

        encoder = StubEncoder()
    else:
        encoder = EnCodecEncoder()

    decoder = BaselineAudioDecoder(
        latent_channels=args.latent_channels,
        out_channels=1,
        base_channels=64,
    )
    if args.decoder and Path(args.decoder).exists():
        decoder.load_state_dict(torch.load(args.decoder, weights_only=False))

    vae = AudioVAE(encoder=encoder, decoder=decoder)
    for p in vae.parameters():
        p.requires_grad_(False)

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
    sched = CosineSchedule(num_steps=args.num_steps)

    print(f"Training DiT for {args.epochs} epochs...")
    losses = list(
        train_audio_diffusion(
            loader,
            vae,
            dit,
            prior,
            sched,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )
    )
    if losses:
        print(f"  Initial loss: {losses[0]['loss']:.4f}")
        print(f"  Final loss:   {losses[-1]['loss']:.4f}")

    torch.save(dit.state_dict(), args.out)
    print(f"Saved DiT to {args.out}")

    from safetensors.torch import save_file

    safe_path = str(Path(args.out).with_suffix(".safetensors"))
    save_file(dit.state_dict(), safe_path)
    print(f"Exported DiT safetensors to {safe_path}")

    config = {
        "dim": args.dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "patch_size": args.patch_size,
        "latent_channels": args.latent_channels,
        "latent_length": latent_length,
        "spec_dim": 3 * args.q,
        "q": args.q,
        "k": 4,
        "num_steps": args.num_steps,
        "audio_length": args.audio_length,
    }
    config_path = Path(args.out).with_suffix("").parent / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    main()
