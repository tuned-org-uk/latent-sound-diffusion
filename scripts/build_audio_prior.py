"""Build the frozen ArrowSpace prior from EnCodec features over a corpus.

Extracts EnCodec encoder features from an audio corpus (ESC-50 or a
generic audio folder), pools them per clip, and builds the ArrowSpace
prior (L_F, U_q, λ_ED) via build_arrow_prior().

Usage:
    uv run python scripts/build_audio_prior.py --data-root /path/to/esc50
    uv run python scripts/build_audio_prior.py --toy --num-samples 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ald_sc.audio_codec import EnCodecEncoder, extract_encodec_features
from ald_sc.build_prior import build_arrow_prior
from ald_sc.data import AudioFolderDataset, Esc50Dataset, ToyAudioDataset, build_audio_dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ArrowSpace prior from audio corpus")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Path to ESC-50 root or audio folder")
    parser.add_argument("--toy", action="store_true",
                        help="Use synthetic ToyAudioDataset (no external data)")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of samples for toy dataset")
    parser.add_argument("--audio-length", type=int, default=24000,
                        help="Audio length in samples (default 24000 = 1s)")
    parser.add_argument("--q", type=int, default=8, help="Spectral modes to retain")
    parser.add_argument("--k", type=int, default=8, help="kNN neighbours")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", type=str, default="prior.pt", help="Output path")
    args = parser.parse_args()

    # Build dataset
    if args.toy or args.data_root is None:
        print("Using ToyAudioDataset (synthetic)")
        dataset = ToyAudioDataset(
            num_samples=args.num_samples,
            audio_length=args.audio_length,
        )
    else:
        data_root = Path(args.data_root)
        if (data_root / "audio").exists():
            print(f"Using Esc50Dataset from {data_root}")
            dataset = Esc50Dataset(root=data_root, audio_length=args.audio_length)
        else:
            print(f"Using AudioFolderDataset from {data_root}")
            dataset = AudioFolderDataset(root=data_root, audio_length=args.audio_length)

    loader = build_audio_dataloader(dataset, batch_size=args.batch_size, shuffle=False)

    # Extract EnCodec features
    print("Loading EnCodec encoder (24 kHz)...")
    encoder = EnCodecEncoder()
    print("Extracting features...")
    embeddings = extract_encodec_features(loader, encoder)
    print(f"Extracted {embeddings.shape[0]} clips, F={embeddings.shape[1]}")

    # Build prior
    print(f"Building ArrowSpace prior (q={args.q}, k={args.k})...")
    prior = build_arrow_prior(embeddings, q=args.q, k=args.k)
    print(f"L_F shape: {prior.L_F.shape}")
    print(f"U_q shape: {prior.U_q.shape}")

    # Save
    torch.save(prior, args.out)
    print(f"Saved prior to {args.out}")


if __name__ == "__main__":
    main()
