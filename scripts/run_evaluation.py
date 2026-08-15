"""Run the full issue #49/#50 evaluation pipeline and produce all deliverables.

Trains the graph + baseline decoders and the 1-D DiT on a real-data subset
(mirroring notebooks/06_end_to_end_real_data.ipynb), then computes every
Phase 1-4 deliverable and writes them to results/.

Framing (issue #50 calibration): the paper presents a novel workflow at
preliminary scale — tables are hypothesis-generating evidence with honest
numbers and documented proxy methodology, not benchmark claims.

Usage:
    # Fast end-to-end self-check (tiny subset, few epochs, CPU/MPS auto)
    uv run python scripts/run_evaluation.py --smoke

    # Full run (256-sample subset, 20 epochs — matches nb06)
    uv run python scripts/run_evaluation.py --device mps

    # Full run + issue #50 Phase 1 sweeps (q / NOISE_INJECT / eps)
    uv run python scripts/run_evaluation.py --device mps \
        --ablation-q 4 8 16 32 --ablation-noise 0.0 0.1 0.25 0.5 \
        --ablation-eps 1e-2 1e-3 1e-4

    # ESC-50 tagged run (does not clobber NSynth tables)
    uv run python scripts/run_evaluation.py --device mps \
        --data-dir data/esc50/ESC-50-master/audio --tag esc50_

Outputs (tag-prefixed):
    results/table1_reconstruction.csv
    results/table2_ablation.csv
    results/table3_fad_clap.csv
    results/table4_compression.csv
    results/table5_coherence.csv
    results/table6_band_retention.csv
    results/recursive_variants.csv
    results/fig_variant_diversity.png
    results/sweep_q.csv  results/sweep_noise.csv  results/sweep_eps.csv
    results/artifacts/  (checkpoints; gitignored)
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
    band_energy_retention,
    compression_ratio_vs_n,
    encodec_pooled_features,
    eps_sweep,
    perceptual_table,
    recursive_variant_drift,
    reconstruction_table,
    rehydration_coherence,
    split_files,
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
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"


def _out(name: str, tag: str = "") -> Path:
    """Tagged results path (empty tag = NSynth default, e.g. 'esc50_' prefix)."""
    return RESULTS_DIR / f"{tag}{name}"


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


def _phase1(encoder, graph_dec, baseline_dec, prior, loss_fn, loaders, device, tag):
    print("\n=== Phase 1: Reconstruction + Ablation + Retention ===")
    rec_rows = reconstruction_table(
        encoder,
        graph_dec,
        baseline_dec,
        prior,
        loss_fn,
        loaders,
        device,
    )
    write_csv(_out("table1_reconstruction.csv", tag), rec_rows)
    print(f"  wrote table1_reconstruction.csv ({len(rec_rows)} rows)")

    abl_rows = ablation_table(
        encoder,
        graph_dec,
        prior,
        loss_fn,
        loaders,
        device,
    )
    write_csv(_out("table2_ablation.csv", tag), abl_rows)
    print(f"  wrote table2_ablation.csv ({len(abl_rows)} rows)")

    ret_rows: list[dict[str, object]] = []
    for split_name, loader in loaders.items():
        for dec_name, dec in (("graph", graph_dec), ("baseline", baseline_dec)):
            rows = band_energy_retention(encoder, dec, prior, loader, device)
            for r in rows:
                r.update(split=split_name, decoder=dec_name)
            ret_rows.extend(rows)
    write_csv(_out("table6_band_retention.csv", tag), ret_rows)
    if ret_rows:
        cos = {(r["split"], r["decoder"]): r["cosine"] for r in ret_rows}
        print(
            f"  wrote table6_band_retention.csv ({len(ret_rows)} rows; "
            f"test cosine graph={cos.get(('test', 'graph')):.4f} "
            f"baseline={cos.get(('test', 'baseline')):.4f})"
        )


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
    write_csv(_out("table3_fad_clap.csv", args.tag), rows)
    print(f"  wrote table3_fad_clap.csv ({len(rows)} rows)")
    for r in rows:
        print(f"    {r['decoder']}: FAD={r['FAD']:.4f} CLAP={r['CLAP']:.4f}")
    return ref_feats


def _phase3a(prior, args):
    print("\n=== Phase 3a: Dehydration compression ratio ===")
    n_values = args.compression_n_values
    rows = compression_ratio_vs_n(n_values, args.audio_length, prior)
    write_csv(_out("table4_compression.csv", args.tag), rows)
    print(f"  wrote table4_compression.csv ({len(rows)} rows)")
    for r in rows[-3:]:
        print(f"    N={r['N']}: ratio={r['compression_ratio']:.2f}x")


def _phase3b(model, bank, tag):
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
    write_csv(_out("table5_coherence.csv", tag), [row])
    print(f"  wrote table5_coherence.csv (pearson_r={row['pearson_r']:.4f})")


def _phase3c(model, encoder, args, device):
    """TRUE recursive variants: condition_on_audio -> synthesize_midi, R rounds."""
    print("\n=== Phase 3c: Recursive variant drift (true recursion) ===")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events = [
        (60, 0.0, 0.4),
        (64, 0.4, 0.4),
        (67, 0.8, 0.4),
    ]
    rounds = max(args.variants_depths)
    rows = recursive_variant_drift(
        model,
        events,
        rounds=rounds,
        encoder=encoder,
        device=device,
        steps=args.steps,
        seed=args.seed,
        bank_n=4,
        strength=0.5,
    )
    write_csv(_out("recursive_variants.csv", args.tag), rows)
    for r in rows:
        print(
            f"    R={r['round']}: dist={r['clap_distance_to_round0']:.5f} "
            f"centroid={r['centroid_mean_hz']:.1f}Hz rolloff={r['rolloff_mean_hz']:.1f}Hz"
        )

    xs = [r["round"] for r in rows]
    dists = [r["clap_distance_to_round0"] for r in rows]
    cents = [r["centroid_mean_hz"] for r in rows]
    rolls = [r["rolloff_mean_hz"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(xs, dists, "o-", color="tab:blue")
    ax1.set_xlabel("Recursion round R")
    ax1.set_ylabel("CLAP-proxy distance to round 0")
    ax1.set_title("Cumulative novelty drift")
    ax1.grid(True, alpha=0.3)
    ax2.plot(xs, cents, "o-", label="centroid (Hz)", color="tab:orange")
    ax2.plot(xs, rolls, "s-", label="rolloff (Hz)", color="tab:green")
    ax2.set_xlabel("Recursion round R")
    ax2.set_ylabel("Mean frequency (Hz)")
    ax2.set_title("Spectral drift over rounds")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = _out("fig_variant_diversity.png", args.tag)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(
        f"  wrote fig_variant_diversity.png (+ {len(rows)} recursive_variants.csv rows)"
    )


# --------------------------------------------------------------------------- #
# Issue #50 Phase 1 sweeps
# --------------------------------------------------------------------------- #


def _train_one_graph_decoder(encoder, prior, loaders, args, device, noise_std, epochs):
    """Train a fresh graph decoder on the given prior; return the decoder."""
    loss_fn = ALDSCLoss(
        prior=prior,
        lambda_rec=1.0,
        lambda_stft=0.0,
        lambda_chart=0.5,
        lambda_smooth=0.1,
    )
    dec = GraphDecoder(
        latent_channels=128,
        out_channels=1,
        feature_dim=128,
        base_channels=args.base_channels,
        prior=prior,
        upsample_strides=(2, 4, 5, 8),
    )
    vae = AudioVAE(encoder=encoder, decoder=dec)
    list(
        log_training(
            train_audio_decoder(
                loaders["train"],
                vae,
                prior,
                loss_fn,
                epochs=epochs,
                lr=args.lr,
                device=device,
                noise_std=noise_std,
            ),
            label=f"Graph decoder (q={prior.q}, noise={noise_std})",
        )
    )
    return dec


def _sweep_q(embeddings, encoder, loaders, args, device):
    """q sweep: rebuild prior per q, retrain graph decoder, test L1 + retention."""
    print(f"\n=== Sweep: q ∈ {args.ablation_q} (decoder retraining) ===")
    rows: list[dict[str, object]] = []
    epochs = args.sweep_decoder_epochs or args.decoder_epochs
    for q in args.ablation_q:
        print(f"  q={q}: building prior + training decoder...")
        prior_q = build_arrow_prior(embeddings, q=q, k=args.k).to(device)
        dec = _train_one_graph_decoder(
            encoder, prior_q, loaders, args, device, args.noise_inject, epochs
        )
        torch.save(dec.state_dict(), ARTIFACTS_DIR / f"{args.tag}graph_dec_q{q}.pt")
        loss_fn_q = ALDSCLoss(
            prior=prior_q,
            lambda_rec=1.0,
            lambda_stft=0.0,
            lambda_chart=0.5,
            lambda_smooth=0.1,
        )
        from ald_sc.eval import evaluate_reconstruction

        m = evaluate_reconstruction(
            encoder, dec, prior_q, loaders["test"], loss_fn_q, device
        )
        ret = band_energy_retention(encoder, dec, prior_q, loaders["test"], device)
        cosine = ret[0]["cosine"] if ret else float("nan")
        rows.append(
            {
                "q": q,
                "test_L1": m["rec"],
                "retention_cosine": cosine,
                "decoder_epochs": epochs,
            }
        )
        print(f"    test L1={m['rec']:.4f} retention_cosine={cosine:.4f}")
    write_csv(_out("sweep_q.csv", args.tag), rows)
    print(f"  wrote sweep_q.csv ({len(rows)} rows)")


def _sweep_noise(prior, encoder, dit, sched, loaders, args, device, main_graph_dec):
    """NOISE_INJECT sweep: fidelity (test L1) vs diversity (variant distance)."""
    print(f"\n=== Sweep: NOISE_INJECT ∈ {args.ablation_noise} ===")
    from ald_sc.eval import evaluate_reconstruction, variant_diversity

    loss_fn = ALDSCLoss(
        prior=prior,
        lambda_rec=1.0,
        lambda_stft=0.0,
        lambda_chart=0.5,
        lambda_smooth=0.1,
    )
    # Fixed conditioning clip from the test split (workflow novelty probe).
    test_clip = next(iter(loaders["test"]))
    test_clip = (test_clip if isinstance(test_clip, torch.Tensor) else test_clip[0])[
        :1
    ].to(device)
    epochs = args.sweep_decoder_epochs or args.decoder_epochs

    rows: list[dict[str, object]] = []
    for noise in args.ablation_noise:
        if abs(noise - args.noise_inject) < 1e-9:
            dec = main_graph_dec  # already trained at this noise level
            reused = True
        else:
            print(f"  noise={noise}: training decoder...")
            dec = _train_one_graph_decoder(
                encoder, prior, loaders, args, device, noise, epochs
            )
            torch.save(
                dec.state_dict(), ARTIFACTS_DIR / f"{args.tag}graph_dec_noise{noise}.pt"
            )
            reused = False
        m = evaluate_reconstruction(
            encoder, dec, prior, loaders["test"], loss_fn, device
        )
        model = LSDModel(
            prior=prior, dit=dit, decoder=dec, encoder=encoder, schedule=sched
        )
        variants = model.condition_on_audio(
            test_clip, n=4, steps=args.steps, seed=args.seed
        )
        div = variant_diversity(lambda d: variants, [len(variants)], encoder, device)[0]
        rows.append(
            {
                "noise_inject": noise,
                "test_L1": m["rec"],
                "variant_distance": div["mean_distance"],
                "reused_main_decoder": reused,
            }
        )
        print(
            f"    noise={noise}: test L1={m['rec']:.4f} "
            f"variant_distance={div['mean_distance']:.5f}"
        )
    write_csv(_out("sweep_noise.csv", args.tag), rows)
    print(f"  wrote sweep_noise.csv ({len(rows)} rows)")


def _sweep_eps(dit, sched, prior, graph_dec, encoder, ref_feats, args, device):
    """Heat-death epsilon sweep (sampling-only)."""
    print(f"\n=== Sweep: eps ∈ {args.ablation_eps} (sampling only) ===")
    rows = eps_sweep(
        dit,
        sched,
        prior,
        graph_dec,
        encoder,
        ref_feats,
        args.ablation_eps,
        device,
        n_gen=args.n_gen,
        steps=args.steps,
        seed=args.seed,
    )
    write_csv(_out("sweep_eps.csv", args.tag), rows)
    for r in rows:
        print(
            f"    eps={r['eps']:g}: mean_steps={r['mean_steps']:.1f} FAD={r['FAD']:.2f}"
        )
    print(f"  wrote sweep_eps.csv ({len(rows)} rows)")


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
        help="Recursive rounds for variant drift (max value = R)",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu | mps | cuda")
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Prefix for output filenames (e.g. 'esc50_'); avoids clobbering",
    )
    parser.add_argument(
        "--ablation-q", type=int, nargs="+", default=[], help="q sweep values"
    )
    parser.add_argument(
        "--ablation-noise",
        type=float,
        nargs="+",
        default=[],
        help="NOISE_INJECT sweep values (diversity-vs-fidelity)",
    )
    parser.add_argument(
        "--ablation-eps",
        type=float,
        nargs="+",
        default=[],
        help="heat-death epsilon sweep values (sampling-only)",
    )
    parser.add_argument(
        "--sweep-decoder-epochs",
        type=int,
        default=None,
        help="Decoder epochs for sweeps (default: --decoder-epochs)",
    )
    args = parser.parse_args()

    if args.smoke:
        args.subset_size = 16
        args.decoder_epochs = 2
        args.diffusion_epochs = 2
        args.n_gen = 4
        args.compression_n_values = [1, 10, 50]
        args.variants_depths = [1, 2, 4]
        args.steps = 20
        args.sweep_decoder_epochs = 1
        args.ablation_q = args.ablation_q[:2]
        args.ablation_noise = args.ablation_noise[:2]
        args.ablation_eps = args.ablation_eps[:2]

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
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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
    torch.save(embeddings, ARTIFACTS_DIR / f"{args.tag}embeddings.pt")
    prior = build_arrow_prior(embeddings, q=args.q, k=args.k).to(device)
    torch.save(prior, ARTIFACTS_DIR / f"{args.tag}prior.pt")
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
    torch.save(graph_dec.state_dict(), ARTIFACTS_DIR / f"{args.tag}graph_dec.pt")
    torch.save(baseline_dec.state_dict(), ARTIFACTS_DIR / f"{args.tag}baseline_dec.pt")
    torch.save(dit.state_dict(), ARTIFACTS_DIR / f"{args.tag}dit.pt")

    _phase1(encoder, graph_dec, baseline_dec, prior, loss_fn, loaders, device, args.tag)
    ref_feats = _phase2(
        encoder, graph_dec, baseline_dec, prior, dit, sched, loaders, device, args
    )
    _phase3a(prior, args)

    model = LSDModel(
        prior=prior, dit=dit, decoder=graph_dec, encoder=encoder, schedule=sched
    )
    bank = model.generate_sound_bank(n=8, steps=args.steps, seed=args.seed)
    bank = [c.cpu() for c in bank]
    _phase3b(model, bank, args.tag)
    _phase3c(model, encoder, args, device)

    if args.ablation_q:
        _sweep_q(embeddings, encoder, loaders, args, device)
    if args.ablation_noise:
        _sweep_noise(prior, encoder, dit, sched, loaders, args, device, graph_dec)
    if args.ablation_eps:
        _sweep_eps(dit, sched, prior, graph_dec, encoder, ref_feats, args, device)

    print("\n=== All deliverables produced ===")
    deliverables = [
        "table1_reconstruction.csv",
        "table2_ablation.csv",
        "table3_fad_clap.csv",
        "table4_compression.csv",
        "table5_coherence.csv",
        "table6_band_retention.csv",
        "recursive_variants.csv",
        "fig_variant_diversity.png",
    ]
    if args.ablation_q:
        deliverables.append("sweep_q.csv")
    if args.ablation_noise:
        deliverables.append("sweep_noise.csv")
    if args.ablation_eps:
        deliverables.append("sweep_eps.csv")
    for f in deliverables:
        p = _out(f, args.tag)
        status = "OK" if p.exists() and p.stat().st_size > 0 else "MISSING"
        print(f"  {status}: results/{args.tag}{f}")


if __name__ == "__main__":
    main()
