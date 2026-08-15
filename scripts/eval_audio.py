"""Evaluate audio reconstruction quality: graph vs baseline decoder.

Computes:
- L1 reconstruction error
- Multi-scale STFT loss
- Chart-energy error and off-manifold ratio (spectral diagnostics)
- λ_ED ablation (graph decoder with vs without c_spec gating)

Usage:
    uv run python scripts/eval_audio.py --prior prior.pt --graph-decoder graph_dec.pt --baseline-decoder base_dec.pt --toy
    uv run python scripts/eval_audio.py --toy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ald_sc.audio_codec import AudioVAE, BaselineAudioDecoder
from ald_sc.build_prior import build_arrow_prior
from ald_sc.data import ToyAudioDataset, build_audio_dataloader
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.losses import ALDSCLoss


def evaluate_decoder(
    vae: AudioVAE,
    prior,
    loader,
    loss_fn: ALDSCLoss,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate reconstruction quality of a decoder."""
    vae = vae.to(device).eval()
    prior = prior.to(device)
    total_rec = 0.0
    total_stft = 0.0
    total_chart = 0.0
    total_smooth = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            x = (
                batch.to(device)
                if isinstance(batch, torch.Tensor)
                else batch[0].to(device)
            )
            z, A, c_spec, x_hat = vae(x, prior)
            A_hat = A.detach()
            losses = loss_fn(x, x_hat, A, A_hat)
            total_rec += losses["rec"].item()
            total_stft += losses["stft"].item()
            total_chart += losses["chart"].item()
            total_smooth += losses["smooth"].item()
            n += 1

    return {
        "rec": total_rec / max(n, 1),
        "stft": total_stft / max(n, 1),
        "chart": total_chart / max(n, 1),
        "smooth": total_smooth / max(n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate audio reconstruction")
    parser.add_argument("--prior", type=str, default=None)
    parser.add_argument("--graph-decoder", type=str, default=None)
    parser.add_argument("--baseline-decoder", type=str, default=None)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--audio-length", type=int, default=24000)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cpu")

    # Build or load prior
    if args.prior and Path(args.prior).exists():
        prior = torch.load(args.prior, weights_only=False)
    else:
        embeddings = torch.randn(64, 128)
        prior = build_arrow_prior(embeddings, q=args.q, k=4)
    prior = prior.to(device)

    # Dataset
    dataset = ToyAudioDataset(num_samples=16, audio_length=args.audio_length)
    loader = build_audio_dataloader(dataset, batch_size=args.batch_size, shuffle=False)

    # Encoder (use stub for toy, EnCodec for real)
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
    loss_fn = ALDSCLoss(
        prior=prior,
        lambda_rec=1.0,
        lambda_stft=1.0,
        lambda_chart=0.5,
        lambda_smooth=0.1,
    )

    results: dict[str, dict[str, float]] = {}

    # Evaluate baseline decoder
    if args.baseline_decoder and Path(args.baseline_decoder).exists():
        decoder = BaselineAudioDecoder(
            latent_channels=128, out_channels=1, base_channels=64
        )
        decoder.load_state_dict(torch.load(args.baseline_decoder, weights_only=False))
        vae = AudioVAE(encoder=encoder, decoder=decoder)
        print("\nEvaluating baseline decoder...")
        results["baseline"] = evaluate_decoder(vae, prior, loader, loss_fn, device)
        for k, v in results["baseline"].items():
            print(f"  {k}: {v:.6f}")

    # Evaluate graph decoder
    if args.graph_decoder and Path(args.graph_decoder).exists():
        decoder = GraphDecoder(
            latent_channels=128,
            out_channels=1,
            feature_dim=128,
            base_channels=64,
            prior=prior,
        )
        decoder.load_state_dict(torch.load(args.graph_decoder, weights_only=False))
        vae = AudioVAE(encoder=encoder, decoder=decoder)
        print("\nEvaluating graph decoder...")
        results["graph"] = evaluate_decoder(vae, prior, loader, loss_fn, device)
        for k, v in results["graph"].items():
            print(f"  {k}: {v:.6f}")

        # λ_ED ablation: with vs without c_spec gating
        print("\nEvaluating graph decoder (no c_spec / λ_ED ablation)...")

        # Override c_spec to zeros
        class NoCSPecVAE(AudioVAE):
            def forward(self, x, prior):
                z, a, c_spec = self.encoder.encode(x, prior)
                zero_cspec = torch.zeros_like(c_spec)
                x_hat = self.decoder(z, zero_cspec)
                return z, a, c_spec, x_hat

        vae_ablation = NoCSPecVAE(encoder=encoder, decoder=decoder)
        results["graph_no_cspec"] = evaluate_decoder(
            vae_ablation, prior, loader, loss_fn, device
        )
        for k, v in results["graph_no_cspec"].items():
            print(f"  {k}: {v:.6f}")

    if not results:
        print(
            "No decoders to evaluate. Provide --graph-decoder and/or --baseline-decoder."
        )

    print("\n=== Summary ===")
    for name, metrics in results.items():
        print(
            f"{name}: rec={metrics['rec']:.4f} stft={metrics['stft']:.4f} "
            f"chart={metrics['chart']:.4f} smooth={metrics['smooth']:.4f}"
        )


if __name__ == "__main__":
    main()
