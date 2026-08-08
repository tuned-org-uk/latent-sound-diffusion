"""CLI image generation for ALD-SC.

Usage:
    uv run python scripts/sample.py --out results/sample.png
    uv run python scripts/sample.py --steps 50 --seed 3407 --out results/sample.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import torch
from ald_sc.build_prior import build_arrow_prior
from ald_sc.data import ToyImageDataset, build_dataloader
from ald_sc.dit import MinimalDiT
from ald_sc.losses import ALDSCLoss
from ald_sc.sampling import sample_ddim
from ald_sc.schedule import CosineSchedule
from ald_sc.trainer import train_vae
from ald_sc.vae import SpectralVAE


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an image with ALD-SC")
    parser.add_argument("--out", type=str, default="results/sample.png", help="Output path")
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--epochs", type=int, default=20, help="VAE training epochs")
    parser.add_argument(
        "--diffusion-epochs", type=int, default=20, help="Diffusion training epochs"
    )
    parser.add_argument("--image-size", type=int, default=32, help="Image size")
    parser.add_argument("--latent-channels", type=int, default=4, help="Latent channels")
    parser.add_argument("--feature-dim", type=int, default=32, help="Feature dimension F")
    parser.add_argument("--q", type=int, default=8, help="Spectral modes")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("Building prior...")
    embeddings = torch.randn(64, args.feature_dim)
    prior = build_arrow_prior(embeddings, q=args.q, k=4)

    print("Creating VAE...")
    vae = SpectralVAE(
        in_channels=3,
        latent_channels=args.latent_channels,
        feature_dim=args.feature_dim,
        base_channels=32,
    )
    loss_fn = ALDSCLoss(prior=prior, lambda_rec=1.0, lambda_chart=0.5, lambda_smooth=0.1)

    print("Training VAE...")
    dataset = ToyImageDataset(num_samples=32, image_size=args.image_size, channels=3)
    loader = build_dataloader(dataset, batch_size=8)
    list(train_vae(loader, vae, prior, loss_fn, epochs=args.epochs, lr=1e-3))

    latent_size = args.image_size // 4
    dit = MinimalDiT(
        latent_channels=args.latent_channels,
        latent_size=latent_size,
        patch_size=2,
        dim=64,
        depth=4,
        num_heads=4,
        text_dim=0,
        spec_dim=3 * args.q,
        cfg_dropout=0.1,
    )

    print("Training DiT...")
    schedule = CosineSchedule(num_steps=1000)
    from ald_sc.trainer import train_diffusion

    list(
        train_diffusion(
            loader,
            vae,
            dit,
            prior,
            schedule,
            epochs=args.diffusion_epochs,
            lr=1e-3,
            cfg_dropout=0.1,
        )
    )

    print("Generating...")
    vae.eval()
    dit.eval()

    c_spec = torch.randn(1, 3 * args.q)
    z = sample_ddim(dit, schedule, c_spec=c_spec, batch_size=1, steps=args.steps, seed=args.seed)

    with torch.no_grad():
        x_hat = vae.decode(z, c_spec, prior)

    img = x_hat[0].permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(out_path), img)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
