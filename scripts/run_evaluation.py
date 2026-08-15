"""Run the full issue #49 evaluation pipeline and produce all deliverables.

Trains the graph + baseline decoders and the 1-D DiT on a real-data subset
(mirroring notebooks/06_end_to_end_real_data.ipynb), then computes every
Phase 1-3 deliverable and writes them to results/.

Usage:
    # Fast end-to-end self-check (tiny subset, few epochs, CPU/MPS auto)
    uv run python scripts/run_evaluation.py --smoke

    # Full run (256-sample subset, 20 epochs — matches nb06)
    uv run python scripts/run_evaluation.py

    # Scaled run on the full corpus
    uv run python scripts/run_evaluation.py --subset-size 0 --decoder-epochs 30

Outputs:
    results/table1_reconstruction.csv
    results/table2_ablation.csv
    results/table3_fad_clap.csv
    results/table4_compression.csv
    results/table5_coherence.csv
    results/fig_variant_diversity.png
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

from torch.utils.data import DataLoader

from ald_sc.audio_codec import AudioVAE, BaselineAudioDecoder, EnCodecEncoder
from ald_sc.build_prior import build_arrow_prior
from ald_sc.data import AudioFolderDataset, build_audio_dataloader
from ald_sc.dit import MinimalDiT
from ald_sc.eval import (
    ablation_table,
    compression_ratio_vs_n,
    encodec_pooled_features,
    perceptual_table,
    reconstruction_table,
    rehydration_coherence,
    split_files,
    variant_diversity,
    write_csv,
)
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.inference import LSDModel
from ald_sc.losses import ALDSCLoss
from ald_sc.schedule import CosineSchedule
from ald_sc.trainer import log_training, train_audio_decoder, train_audio_diffusion

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


def _device(device_arg: str | None = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_loaders(
    train_files, val_files, test_files, audio_length, sample_rate, batch_size, data_dir
):
    def _ds(files):
        ds = AudioFolderDataset(
            root=str(data_dir), audio_length=audio_length, sample_rate=sample_rate
        )
        ds.files = files
        return ds

    train_ds = _ds(train_files)
    val_ds = _ds(val_files)
    test_ds = _ds(test_files)
    # Training loader keeps drop_last=True (matches nb06); val/test use
    # drop_last=False so small splits still produce batches for evaluation.
    loaders = {
        "train": build_audio_dataloader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(
            val_ds, batch_size=min(batch_size, max(len(val_ds), 1)), shuffle=False
        ),
        "test": DataLoader(
            test_ds, batch_size=min(batch_size, max(len(test_ds), 1)), shuffle=False
        ),
    }
    return loaders, train_ds, val_ds, test_ds


def _train_models(encoder, prior, loaders, args, device):
    loss_fn = ALDSCLoss(
        prior=prior,
        lambda_rec=1.0,
        lambda_stft=0.0,
        lambda_chart=0.5,
        lambda_smooth=0.1,
    )
    graph_dec = GraphDecoder(
        latent_channels=128,
        out_channels=1,
        feature_dim=128,
        base_channels=args.base_channels,
        prior=prior,
        upsample_strides=(2, 4, 5, 8),
    )
    baseline_dec = BaselineAudioDecoder(
        latent_channels=128,
        out_channels=1,
        base_channels=args.base_channels,
        upsample_strides=(2, 4, 5, 8),
    )
    graph_vae = AudioVAE(encoder=encoder, decoder=graph_dec)
    baseline_vae = AudioVAE(encoder=encoder, decoder=baseline_dec)

    print("Training graph decoder...")
    t0 = time.time()
    list(
        log_training(
            train_audio_decoder(
                loaders["train"],
                graph_vae,
                prior,
                loss_fn,
                epochs=args.decoder_epochs,
                lr=args.lr,
                device=device,
                noise_std=args.noise_inject,
            ),
            label="Graph decoder",
        )
    )
    print(f"  done in {time.time() - t0:.1f}s")

    print("Training baseline decoder...")
    t0 = time.time()
    list(
        log_training(
            train_audio_decoder(
                loaders["train"],
                baseline_vae,
                prior,
                loss_fn,
                epochs=args.decoder_epochs,
                lr=args.lr,
                device=device,
                noise_std=args.noise_inject,
            ),
            label="Baseline decoder",
        )
    )
    print(f"  done in {time.time() - t0:.1f}s")

    latent_length = args.audio_length // 320
    dit = MinimalDiT(
        latent_channels=128,
        latent_length=latent_length,
        patch_size=8,
        dim=64,
        depth=2,
        num_heads=4,
        spec_dim=3 * args.q,
    )
    sched = CosineSchedule(num_steps=1000)

    print("Training 1-D DiT...")
    t0 = time.time()
    for p in graph_vae.parameters():
        p.requires_grad_(False)
    list(
        log_training(
            train_audio_diffusion(
                loaders["train"],
                graph_vae,
                dit,
                prior,
                sched,
                epochs=args.diffusion_epochs,
                lr=args.lr,
                device=device,
            ),
            label="DiT",
        )
    )
    print(f"  done in {time.time() - t0:.1f}s")

    return graph_dec, baseline_dec, dit, sched, loss_fn


def _phase1(encoder, graph_dec, baseline_dec, prior, loss_fn, loaders, device):
    print("\n=== Phase 1: Reconstruction + Ablation ===")
    rec_rows = reconstruction_table(
        encoder,
        graph_dec,
        baseline_dec,
        prior,
        loss_fn,
        loaders,
        device,
    )
    write_csv(RESULTS_DIR / "table1_reconstruction.csv", rec_rows)
    print(f"  wrote results/table1_reconstruction.csv ({len(rec_rows)} rows)")

    abl_rows = ablation_table(
        encoder,
        graph_dec,
        prior,
        loss_fn,
        loaders,
        device,
    )
    write_csv(RESULTS_DIR / "table2_ablation.csv", abl_rows)
    print(f"  wrote results/table2_ablation.csv ({len(abl_rows)} rows)")


def _phase2(encoder, graph_dec, baseline_dec, prior, dit, sched, loaders, device, args):
    print("\n=== Phase 2: FAD-proxy + CLAP-proxy ===")
    # Use the test loader (already correctly sized for small splits).
    clips_iter = (b if isinstance(b, torch.Tensor) else b[0] for b in loaders["test"])
    ref_feats = encodec_pooled_features(clips_iter, encoder, device)
    if ref_feats.shape[0] == 0:
        # Fallback: iterate the test dataset point-by-point.
        test_ds = loaders["test"].dataset
        ref_feats = encodec_pooled_features(
            [test_ds[i] for i in range(len(test_ds))],
            encoder,
            device,
        )
    rows = perceptual_table(
        encoder,
        graph_dec,
        baseline_dec,
        prior,
        dit,
        sched,
        ref_feats,
        args.text_prompt,
        device,
        n_gen=args.n_gen,
        steps=args.steps,
        seed=args.seed,
    )
    for r in rows:
        r["text_prompt"] = args.text_prompt
        r["n_gen"] = args.n_gen
    write_csv(RESULTS_DIR / "table3_fad_clap.csv", rows)
    print(f"  wrote results/table3_fad_clap.csv ({len(rows)} rows)")
    for r in rows:
        print(f"    {r['decoder']}: FAD={r['FAD']:.4f} CLAP={r['CLAP']:.4f}")


def _phase3a(prior, args):
    print("\n=== Phase 3a: Dehydration compression ratio ===")
    n_values = args.compression_n_values
    rows = compression_ratio_vs_n(n_values, args.audio_length, prior)
    write_csv(RESULTS_DIR / "table4_compression.csv", rows)
    print(f"  wrote results/table4_compression.csv ({len(rows)} rows)")
    for r in rows[-3:]:
        print(f"    N={r['N']}: ratio={r['compression_ratio']:.2f}x")


def _phase3b(model, bank):
    print("\n=== Phase 3b: Rehydration coherence ===")
    events = [
        (60, 0.0, 0.5),
        (62, 0.5, 0.5),
        (64, 1.0, 0.5),
        (65, 1.5, 0.5),
        (67, 2.0, 0.5),
        (69, 2.5, 0.5),
    ]
    row = rehydration_coherence(model, events, bank, pitch_bank_root=60)
    row["midi_seq"] = "ascending_scale"
    write_csv(RESULTS_DIR / "table5_coherence.csv", [row])
    print(f"  wrote results/table5_coherence.csv (pearson_r={row['pearson_r']:.4f})")


def _phase3c(encoder, prior, dit, sched, graph_dec, device, args):
    print("\n=== Phase 3c: Recursive variant diversity ===")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = LSDModel(
        prior=prior,
        dit=dit,
        decoder=graph_dec,
        encoder=encoder,
        schedule=sched,
    )
    depths = args.variants_depths

    midi_seqs = [
        [(60, 0.0, 0.4), (64, 0.4, 0.4), (67, 0.8, 0.4)],
        [(62, 0.0, 0.4), (65, 0.4, 0.4), (69, 0.8, 0.4)],
        [(55, 0.0, 0.5), (59, 0.5, 0.5), (62, 1.0, 0.5)],
        [(48, 0.0, 0.6), (52, 0.6, 0.6), (55, 1.2, 0.6)],
        [(72, 0.0, 0.3), (76, 0.3, 0.3), (79, 0.6, 0.3)],
        [(57, 0.0, 0.4), (60, 0.4, 0.4), (64, 0.8, 0.4)],
        [(65, 0.0, 0.5), (69, 0.5, 0.5), (72, 1.0, 0.5)],
        [(50, 0.0, 0.4), (53, 0.4, 0.4), (57, 0.8, 0.4)],
    ]
    max_depth = max(depths)
    print(f"  Generating shared bank of {max_depth} sounds (steps={args.steps})...")
    shared_bank = model.generate_sound_bank(
        n=max_depth, steps=args.steps, seed=args.seed
    )
    # Free MPS memory by moving bank clips to CPU before MIDI rendering
    # (resampling is done on CPU).
    shared_bank = [c.cpu() for c in shared_bank]

    def make_variants(d):
        clips = []
        for i in range(d):
            seq = midi_seqs[i % len(midi_seqs)]
            audio = model.synthesize_midi(
                seq, shared_bank, pitch_bank_root=60, seed=args.seed + i
            )
            clips.append(audio.unsqueeze(0).cpu())
        return clips

    rows = variant_diversity(make_variants, depths, encoder, device)
    write_csv(RESULTS_DIR / "variant_diversity.csv", rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [r["depth"] for r in rows]
    means = [r["mean_distance"] for r in rows]
    mins = [r["min_distance"] for r in rows]
    maxs = [r["max_distance"] for r in rows]
    ax.plot(xs, means, "o-", label="mean pairwise distance", color="tab:blue")
    ax.fill_between(xs, mins, maxs, alpha=0.2, color="tab:blue")
    ax.set_xlabel("Recursion depth R")
    ax.set_ylabel("CLAP-proxy cosine distance")
    ax.set_title("Recursive variant diversity vs depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = RESULTS_DIR / "fig_variant_diversity.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  wrote results/fig_variant_diversity.png (and {len(rows)} CSV rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the issue #49 evaluation pipeline"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Fast end-to-end self-check"
    )
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--subset-size", type=int, default=256, help="0 = full corpus")
    parser.add_argument("--audio-length", type=int, default=96000)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--decoder-epochs", type=int, default=20)
    parser.add_argument("--diffusion-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-inject", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n-gen", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--text-prompt", type=str, default="warm electronic bass synth")
    parser.add_argument(
        "--compression-n-values",
        type=int,
        nargs="+",
        default=[1, 10, 50, 100, 256, 1000, 5000, 9811],
    )
    parser.add_argument(
        "--variants-depths",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Recursion depths for variant diversity (issue #49: 1x, 2x, 4x)",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu | mps | cuda")
    args = parser.parse_args()

    if args.smoke:
        args.subset_size = 16
        args.decoder_epochs = 2
        args.diffusion_epochs = 2
        args.n_gen = 4
        args.compression_n_values = [1, 10, 50]
        args.variants_depths = [1, 2, 4]
        args.steps = 20

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    all_files = sorted(data_dir.glob("*.wav"))
    if not all_files:
        print(f"No .wav files in {data_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(all_files)} audio files in {data_dir}")

    if args.subset_size and args.subset_size > 0 and len(all_files) > args.subset_size:
        rng = random.Random(args.seed)
        selected = rng.sample(all_files, args.subset_size)
    else:
        selected = all_files

    train_files, val_files, test_files = split_files(
        selected,
        args.train_frac,
        args.val_frac,
        seed=args.seed,
    )
    print(
        f"Split: train={len(train_files)} val={len(val_files)} test={len(test_files)}"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    encoder = EnCodecEncoder(sample_rate=args.sample_rate, bandwidth=24)

    print("Building prior from training-set EnCodec features...")
    prior_ds = AudioFolderDataset(
        root=str(data_dir),
        audio_length=args.audio_length,
        sample_rate=args.sample_rate,
    )
    prior_ds.files = train_files
    prior_loader = build_audio_dataloader(
        prior_ds, batch_size=args.batch_size, shuffle=False
    )
    feats = []
    for batch in prior_loader:
        z = encoder.extract_features(batch)
        feats.append(z.mean(dim=2))
    embeddings = torch.cat(feats, dim=0).cpu()
    prior = build_arrow_prior(embeddings, q=args.q, k=args.k).to(device)
    print(f"Prior: L_F={prior.L_F.shape}, U_q={prior.U_q.shape}, q={prior.q}")

    loaders, train_ds, val_ds, test_ds = _build_loaders(
        train_files,
        val_files,
        test_files,
        args.audio_length,
        args.sample_rate,
        args.batch_size,
        data_dir,
    )

    graph_dec, baseline_dec, dit, sched, loss_fn = _train_models(
        encoder,
        prior,
        loaders,
        args,
        device,
    )

    _phase1(encoder, graph_dec, baseline_dec, prior, loss_fn, loaders, device)
    _phase2(encoder, graph_dec, baseline_dec, prior, dit, sched, loaders, device, args)
    _phase3a(prior, args)

    model = LSDModel(
        prior=prior, dit=dit, decoder=graph_dec, encoder=encoder, schedule=sched
    )
    bank = model.generate_sound_bank(n=8, steps=args.steps, seed=args.seed)
    bank = [c.cpu() for c in bank]
    _phase3b(model, bank)
    _phase3c(encoder, prior, dit, sched, graph_dec, device, args)

    print("\n=== All deliverables produced ===")
    for f in [
        "table1_reconstruction.csv",
        "table2_ablation.csv",
        "table3_fad_clap.csv",
        "table4_compression.csv",
        "table5_coherence.csv",
        "fig_variant_diversity.png",
    ]:
        p = RESULTS_DIR / f
        status = "OK" if p.exists() and p.stat().st_size > 0 else "MISSING"
        print(f"  {status}: results/{f}")


if __name__ == "__main__":
    main()
