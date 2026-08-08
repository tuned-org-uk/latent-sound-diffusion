"""Train the audio decoder (graph or baseline) with frozen EnCodec encoder.

Usage:
    uv run python scripts/train_audio_decoder.py --prior prior.pt --graph --epochs 50
    uv run python scripts/train_audio_decoder.py --prior prior.pt --baseline --epochs 50
    uv run python scripts/train_audio_decoder.py --toy --epochs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ald_sc.audio_codec import AudioVAE, BaselineAudioDecoder, EnCodecEncoder
from ald_sc.build_prior import build_arrow_prior
from ald_sc.data import AudioFolderDataset, Esc50Dataset, ToyAudioDataset, build_audio_dataloader
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.losses import ALDSCLoss
from ald_sc.trainer import train_audio_decoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train audio decoder")
    parser.add_argument("--prior", type=str, default=None, help="Path to saved prior")
    parser.add_argument("--graph", action="store_true", help="Use graph decoder")
    parser.add_argument("--baseline", action="store_true", help="Use baseline decoder")
    parser.add_argument("--toy", action="store_true", help="Use ToyAudioDataset")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--audio-length", type=int, default=24000)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="decoder.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build or load prior
    if args.prior and Path(args.prior).exists():
        prior = torch.load(args.prior, weights_only=False)
    else:
        print("Building toy prior (no prior file found)")
        embeddings = torch.randn(64, args.feature_dim)
        prior = build_arrow_prior(embeddings, q=args.q, k=4)
    prior = prior.to(device)

    # Build dataset
    if args.toy or args.data_root is None:
        dataset = ToyAudioDataset(num_samples=32, audio_length=args.audio_length)
    else:
        data_root = Path(args.data_root)
        if (data_root / "audio").exists():
            dataset = Esc50Dataset(root=data_root, audio_length=args.audio_length)
        else:
            dataset = AudioFolderDataset(root=data_root, audio_length=args.audio_length)

    loader = build_audio_dataloader(dataset, batch_size=args.batch_size, shuffle=True)

    # Build encoder
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

    # Build decoder
    if args.graph:
        decoder = GraphDecoder(
            latent_channels=args.latent_channels,
            out_channels=1,
            feature_dim=args.feature_dim,
            base_channels=args.base_channels,
            prior=prior,
        )
        decoder_type = "graph"
    else:
        decoder = BaselineAudioDecoder(
            latent_channels=args.latent_channels,
            out_channels=1,
            base_channels=args.base_channels,
        )
        decoder_type = "baseline"

    vae = AudioVAE(encoder=encoder, decoder=decoder)
    loss_fn = ALDSCLoss(
        prior=prior,
        lambda_rec=1.0,
        lambda_stft=1.0,
        lambda_chart=0.5,
        lambda_smooth=0.1,
    )

    print(f"Training {decoder_type} decoder for {args.epochs} epochs...")
    losses = list(train_audio_decoder(loader, vae, prior, loss_fn,
                                       epochs=args.epochs, lr=args.lr, device=device))
    if losses:
        print(f"  Initial loss: {losses[0]['loss']:.4f}")
        print(f"  Final loss:   {losses[-1]['loss']:.4f}")

    torch.save(decoder.state_dict(), args.out)
    print(f"Saved decoder to {args.out}")

    from safetensors.torch import save_file

    safe_path = str(Path(args.out).with_suffix(".safetensors"))
    save_file(decoder.state_dict(), safe_path)
    print(f"Exported decoder safetensors to {safe_path}")

    config = {
        "decoder_type": decoder_type,
        "latent_channels": args.latent_channels,
        "out_channels": 1,
        "feature_dim": args.feature_dim,
        "base_channels": args.base_channels,
        "q": args.q,
        "k": 4,
        "upsample_strides": [2, 4, 5, 8],
        "audio_length": args.audio_length,
        "sample_rate": 24000,
    }
    config_path = Path(args.out).with_suffix("").parent / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    main()
