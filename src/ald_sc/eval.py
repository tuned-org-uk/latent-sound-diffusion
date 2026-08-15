"""Evaluation pipeline for the Spectral Composition paper (issue #49).

Owns all paper-table computations:

- Phase 1: train/val/test L1 reconstruction + lambda_ED ablation
  (``reconstruction_table``, ``ablation_table``).
- Phase 2: FAD-proxy (Frechet distance over frozen EnCodec feature
  distributions) + CLAP-proxy (cosine on a deterministic text/audio
  embedding) + baseline comparison (``frechet_distance``, ``fad_score``,
  ``text_embedding``, ``audio_embedding``, ``clap_proxy_score``).
- Phase 3a: dehydration compression ratio vs corpus size N
  (``compression_ratio``, ``compression_ratio_vs_n``).
- Phase 3b: rehydration coherence (MIDI pitch contour vs spectral
  centroid, Pearson r) (``rehydration_coherence``).
- Phase 3c: recursive variant diversity (pairwise CLAP-proxy cosine
  distance vs depth) (``variant_diversity``).

Dependency note. ``fadtk`` and ``laion-clap`` were removed from the project
due to dependency conflicts (see README "Open issues"). To stay green, the
Phase 2 / 3c "FAD" and "CLAP score" columns are implemented as faithful,
dependency-free proxies on the *same* frozen EnCodec feature space the model
already uses. The methodology is documented alongside each table so the
proxy columns are never mistaken for the original VGGish / LAION metrics.

This module must not define model architectures or training loops (per
AGENTS.md S11); it only evaluates already-trained components.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.audio_codec import BaselineAudioDecoder, EnCodecEncoder
from ald_sc.losses import ALDSCLoss

__all__ = [
    "split_files",
    "evaluate_reconstruction",
    "reconstruction_table",
    "ablation_table",
    "frechet_distance",
    "encodec_pooled_features",
    "fad_score",
    "text_embedding",
    "audio_embedding",
    "clap_proxy_score",
    "perceptual_table",
    "compression_ratio",
    "compression_ratio_vs_n",
    "spectral_centroid",
    "midi_pitch_contour",
    "rehydration_coherence",
    "variant_diversity",
]

SAMPLE_RATE = 24000


def split_files(
    files: Sequence[str | Path],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 3407,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Reproducible train/val/test split of a file list (mirrors nb06).

    The remainder after ``train_frac + val_frac`` goes to test. The split is
    seeded and the input list is sorted before shuffling so it is deterministic
    across runs and independent of the OS file order.
    """
    import random

    ordered = sorted(Path(f) for f in files)
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


@torch.no_grad()
def evaluate_reconstruction(
    encoder: EnCodecEncoder,
    decoder: nn.Module,
    prior: ArrowSpacePrior,
    loader,
    loss_fn: ALDSCLoss,
    device: torch.device,
    use_cspec: bool = True,
) -> dict[str, float]:
    """Mean reconstruction metrics of a decoder over a loader.

    Handles both graph decoders (which take ``(z, c_spec)``) and the matched
    baseline (which takes ``z`` only). When ``use_cspec`` is False the graph
    decoder receives a zeroed ``c_spec`` — the lambda_ED ablation.
    """
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    prior = prior.to(device)
    is_baseline = isinstance(decoder, BaselineAudioDecoder)

    totals = {"rec": 0.0, "stft": 0.0, "chart": 0.0, "smooth": 0.0}
    n = 0
    for batch in loader:
        x = batch.to(device) if isinstance(batch, Tensor) else batch[0].to(device)
        z, A, c_spec = encoder.encode(x, prior)
        if is_baseline:
            x_hat = decoder(z)
        elif use_cspec:
            x_hat = decoder(z, c_spec)
        else:
            x_hat = decoder(z, torch.zeros_like(c_spec))
        losses = loss_fn(x, x_hat, A, A.detach())
        totals["rec"] += float(losses["rec"].item())
        totals["stft"] += float(losses["stft"].item())
        totals["chart"] += float(losses["chart"].item())
        totals["smooth"] += float(losses["smooth"].item())
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def reconstruction_table(
    encoder: EnCodecEncoder,
    graph_decoder: nn.Module,
    baseline_decoder: nn.Module,
    prior: ArrowSpacePrior,
    loss_fn: ALDSCLoss,
    loaders: dict[str, object],
    device: torch.device,
) -> list[dict[str, object]]:
    """Phase 1, Table 1: mean L1 reconstruction per split per decoder.

    Returns one row per (split, decoder) with columns
    ``split, decoder, L1, stft, chart, smooth``.
    """
    rows: list[dict[str, object]] = []
    for split_name, loader in loaders.items():
        for dec_name, dec in (("graph", graph_decoder), ("baseline", baseline_decoder)):
            m = evaluate_reconstruction(encoder, dec, prior, loader, loss_fn, device)
            rows.append(
                {
                    "split": split_name,
                    "decoder": dec_name,
                    "L1": m["rec"],
                    "stft": m["stft"],
                    "chart": m["chart"],
                    "smooth": m["smooth"],
                }
            )
    return rows


def ablation_table(
    encoder: EnCodecEncoder,
    graph_decoder: nn.Module,
    prior: ArrowSpacePrior,
    loss_fn: ALDSCLoss,
    loaders: dict[str, object],
    device: torch.device,
) -> list[dict[str, object]]:
    """Phase 1, Table 2: lambda_ED ablation (c_spec on vs off).

    For each split reports the graph-decoder L1 with and without c_spec
    gating, plus the lambda_ED delta (without - with; negative means gating
    helps reconstruction). Columns: ``split, with_cspec, L1`` and a final
    delta row per split.
    """
    rows: list[dict[str, object]] = []
    for split_name, loader in loaders.items():
        with_c = evaluate_reconstruction(
            encoder, graph_decoder, prior, loader, loss_fn, device, use_cspec=True
        )
        without = evaluate_reconstruction(
            encoder, graph_decoder, prior, loader, loss_fn, device, use_cspec=False
        )
        rows.append({"split": split_name, "with_cspec": True, "L1": with_c["rec"]})
        rows.append({"split": split_name, "with_cspec": False, "L1": without["rec"]})
        rows.append(
            {
                "split": split_name,
                "with_cspec": "delta",
                "L1": without["rec"] - with_c["rec"],
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Phase 2: perceptual metrics (dependency-free proxies)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def encodec_pooled_features(
    audio_iter: Iterable[Tensor],
    encoder: EnCodecEncoder,
    device: torch.device,
) -> Tensor:
    """Concatenate pooled EnCodec features (N, F) for an iterable of clips.

    ``audio_iter`` yields waveforms shaped (B, 1, T_audio) or (B, T_audio) or
    (1, 1, T_audio). Batched inputs produce multiple feature vectors.
    Returns a single (N, F) tensor where N is the total number of clips.
    """
    encoder = encoder.to(device).eval()
    feats: list[Tensor] = []
    for clip in audio_iter:
        x = clip.to(device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() == 3 and x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        z = encoder.extract_features(x)
        feats.append(z.mean(dim=2).cpu())
    return torch.cat(feats, dim=0)


def frechet_distance(feats_a: Tensor, feats_b: Tensor) -> float:
    """Frechet distance between two feature distributions.

    ``||mu_a - mu_b||^2 + Tr(C_a + C_b - 2 (C_a C_b)^{1/2})`` on per-sample
    feature vectors. This is the Frechet Audio Distance formula evaluated on
    EnCodec pooled features (the FAD-proxy used here).
    """
    a = feats_a.detach().float()
    b = feats_b.detach().float()
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("frechet_distance expects (N, F) feature matrices")
    mu_a = a.mean(dim=0)
    mu_b = b.mean(dim=0)
    ca = torch.cov(a.T)
    cb = torch.cov(b.T)
    diff = (mu_a - mu_b).pow(2).sum().item()
    # eigenvalues of (Ca @ Cb) are real nonneg for PSD factors; sqrt-sum is
    # Tr(sqrt(Ca Cb)). Guard with symmetric sqrt via Ca^(1/2) Cb Ca^(1/2).
    eigvals_a, eigvecs_a = torch.linalg.eigh(ca)
    eigvals_a = eigvals_a.clamp(min=0)
    sqrt_ca = (eigvecs_a * eigvals_a.sqrt().unsqueeze(0)) @ eigvecs_a.T
    mid = sqrt_ca @ cb @ sqrt_ca
    mid = (mid + mid.T) / 2
    eigvals_mid = torch.linalg.eigvalsh(mid).clamp(min=0)
    trace_term = float(eigvals_mid.sqrt().sum().item())
    ca_reg = float(torch.trace(ca).item())
    cb_reg = float(torch.trace(cb).item())
    return float(diff + ca_reg + cb_reg - 2.0 * trace_term)


def fad_score(generated_features: Tensor, reference_features: Tensor) -> float:
    """FAD-proxy: Frechet distance between generated and reference feature sets.

    Lower is better (generated distribution closer to the reference corpus).
    Falls back to a simple ||mu|| difference when either set has < 2 samples
    (a covariance needs at least 2 points).
    """
    if generated_features.shape[0] < 2 or reference_features.shape[0] < 2:
        return float(
            (generated_features.mean(0) - reference_features.mean(0))
            .pow(2)
            .sum()
            .item()
        )
    val = frechet_distance(generated_features, reference_features)
    return max(val, 0.0)


def text_embedding(text: str, dim: int, salt: int = 0) -> Tensor:
    """Deterministic text -> dim embedding (CLAP-proxy text side).

    A hashing bag-of-characters projection that is fully determined by the
    string (and an optional salt). The result is L2-normalised. This is *not*
    a learned text encoder; it gives a stable, reproducible target vector so
    the cosine-alignment score is well defined and responds to the prompt.
    """
    vec = torch.zeros(dim, dtype=torch.float32)
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).digest()
    # Walk the hash bytes, scattering signed contributions across dim. Iterate
    # the digest cyclically so the vector is dense for short strings.
    i = 0
    while vec.abs().sum().item() == 0 or i < max(8, dim):
        chunk = hashlib.sha256(digest + i.to_bytes(4, "little")).digest()
        for byte in chunk:
            sign = 1.0 if (byte & 1) else -1.0
            mag = ((byte >> 1) & 0x7F) / 127.0
            vec[(i + (byte >> 1)) % dim] += sign * mag
            i += 1
            if vec.abs().sum().item() > 0 and i >= max(8, dim):
                break
        if i >= max(256, dim * 4):
            break
    norm = vec.norm()
    if norm > 0:
        vec = vec / norm
    return vec


@torch.no_grad()
def audio_embedding(
    audio: Tensor | Iterable[Tensor],
    encoder: EnCodecEncoder,
    prior: ArrowSpacePrior | None,
    device: torch.device,
) -> Tensor:
    """CLAP-proxy audio side: pooled EnCodec features of one or many clips.

    Accepts a single (1, 1, T) clip or an iterable of clips and returns a
    single (F,) mean feature (for a single clip) or (N, F) features. Normalised
    to unit norm so cosine similarity is well scaled.
    """
    if isinstance(audio, Tensor):
        audio = [audio]
    feats = encodec_pooled_features(audio, encoder, device)
    if feats.shape[0] == 1:
        v = feats[0]
    else:
        v = feats.mean(dim=0)
    norm = v.norm()
    if norm > 0:
        v = v / norm
    return v


def clap_proxy_score(
    generated_audio: list[Tensor],
    text_prompt: str,
    encoder: EnCodecEncoder,
    device: torch.device,
) -> float:
    """Mean cosine similarity between generated clips and a text prompt.

    Audio side = L2-normalised mean EnCodec-pooled feature of the generated
    bank; text side = L2-normalised deterministic hashing embedding of the
    prompt into the same F dimensions. Returns a scalar in [-1, 1]. This is the
    documented CLAP-proxy (not LAION-CLAP).
    """
    audio_vec = audio_embedding(generated_audio, encoder, None, device)
    feat_dim = audio_vec.shape[0]
    text_vec = text_embedding(text_prompt, feat_dim).to(audio_vec.device)
    return float(torch.cosine_similarity(audio_vec, text_vec, dim=0).item())


def perceptual_table(
    encoder: EnCodecEncoder,
    graph_decoder: nn.Module,
    baseline_decoder: nn.Module,
    prior: ArrowSpacePrior,
    dit: nn.Module,
    sched,
    reference_features: Tensor,
    text_prompt: str,
    device: torch.device,
    n_gen: int = 16,
    steps: int = 50,
    seed: int = 3407,
) -> list[dict[str, object]]:
    """Phase 2, Table 3: FAD-proxy + CLAP-proxy for graph vs baseline.

    Generates ``n_gen`` latents via DDIM, decodes with each decoder, and
    compares the resulting audio feature distribution to ``reference_features``
    (the held-out test-set EnCodec features). Also reports the CLAP-proxy
    alignment between the generated bank and ``text_prompt``.
    """
    from ald_sc.sampling import sample_ddim

    rows: list[dict[str, object]] = []
    encoder = encoder.to(device).eval()
    prior = prior.to(device)
    dit = dit.to(device).eval()

    def _gen_audio(decoder, use_cspec: bool) -> list[Tensor]:
        clips: list[Tensor] = []
        for i in range(n_gen):
            z = sample_ddim(
                dit,
                sched,
                batch_size=1,
                steps=steps,
                seed=seed + i,
                device=device,
            )
            a = z.mean(dim=2)
            c_spec = prior.chart_energy_descriptor(a)
            if isinstance(decoder, BaselineAudioDecoder):
                x_hat = decoder(z)
            elif use_cspec:
                x_hat = decoder(z, c_spec)
            else:
                x_hat = decoder(z, torch.zeros_like(c_spec))
            x_hat = x_hat.clamp(-1, 1).squeeze(0)
            peak = x_hat.abs().max()
            if peak > 0:
                x_hat = x_hat / peak
            clips.append(x_hat.unsqueeze(0))
        return clips

    for name, dec, use_c in (
        ("graph", graph_decoder, True),
        ("baseline", baseline_decoder, False),
    ):
        clips = _gen_audio(dec, use_c)
        gen_feats = encodec_pooled_features(clips, encoder, device)
        # Both feature sets are on CPU (encodec_pooled_features moves to CPU).
        fad = fad_score(gen_feats, reference_features.cpu())
        clap = clap_proxy_score(clips, text_prompt, encoder, device)
        rows.append(
            {
                "decoder": name,
                "FAD": fad,
                "FAD_method": "encod_proxy",
                "CLAP": clap,
                "CLAP_method": "encod_proxy",
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Phase 3a: dehydration compression ratio
# --------------------------------------------------------------------------- #


def _prior_float_count(prior: ArrowSpacePrior) -> int:
    """Number of float32 values stored in the dehydrated prior."""
    n = 0
    for buf_name in ("L_F", "U_q", "eigvals_q", "lambdas_ed", "lambdas_chart"):
        buf = getattr(prior, buf_name, None)
        if buf is not None:
            n += int(buf.numel())
    return n


def compression_ratio(
    n_clips: int,
    audio_length: int,
    prior: ArrowSpacePrior,
    bits_per_pcm_sample: int = 16,
    bits_per_latent_element: int = 16,
    latent_length: int | None = None,
    latent_channels: int = 128,
    encod_bandwidth_kbps: int = 24,
    use_code_rate: bool = True,
) -> dict[str, float]:
    """Dehydration compression ratio for a corpus of ``n_clips``.

    Two representations are compared:

    - **Raw audio**: ``n_clips * audio_length * bits_per_pcm_sample`` bits
      (16-bit PCM library).
    - **Dehydrated**: the shared ArrowSpace prior
      ``(L_F, U_q, eigvals_q, lambda_ED, lambda_chart)`` stored once (in
      float32 = 32 bits/value) plus per-clip latent storage. The latent
      storage models EnCodec's actual compressed rate:
      ``encod_bandwidth_kbps * (audio_length / sample_rate)`` bits/clip. Set
      ``use_code_rate=False`` to instead count latent tensors as
      ``latent_channels * latent_length * bits_per_latent_element``.

    Returns raw/dehydrated bit counts and the ratio raw/dehydrated. The
    prior is constant across N, so the ratio grows with N and asymptotes to
    the per-clip code rate ratio (the prior amortises).
    """
    raw_bits = float(n_clips * audio_length * bits_per_pcm_sample)
    prior_bits = float(_prior_float_count(prior) * 32)
    if use_code_rate:
        latent_bits_per_clip = float(
            encod_bandwidth_kbps * 1000 * (audio_length / SAMPLE_RATE)
        )
    else:
        if latent_length is None:
            latent_length = audio_length // 320
        latent_bits_per_clip = float(
            latent_channels * latent_length * bits_per_latent_element
        )
    dehydrated_bits = prior_bits + n_clips * latent_bits_per_clip
    ratio = raw_bits / dehydrated_bits if dehydrated_bits > 0 else float("inf")
    return {
        "N": n_clips,
        "raw_bits": raw_bits,
        "prior_bits": prior_bits,
        "latent_bits_per_clip": latent_bits_per_clip,
        "dehydrated_bits": dehydrated_bits,
        "compression_ratio": ratio,
    }


def compression_ratio_vs_n(
    n_values: Sequence[int],
    audio_length: int,
    prior: ArrowSpacePrior,
    **kwargs,
) -> list[dict[str, float]]:
    """Compression ratio as a function of corpus size N (Table 4 rows)."""
    return [compression_ratio(n, audio_length, prior, **kwargs) for n in n_values]


# --------------------------------------------------------------------------- #
# Phase 3b: rehydration coherence
# --------------------------------------------------------------------------- #


def spectral_centroid(
    audio: Tensor, sample_rate: int = SAMPLE_RATE, n_fft: int = 2048, hop: int = 512
) -> Tensor:
    """Per-frame spectral centroid (1-D tensor) of a mono waveform.

    centroid(t) = sum_f f * |X(t,f)| / sum_f |X(t,f)|.

    Accepts (T,) or (1, T) or (1, 1, T). Returns (num_frames,) centroid in Hz.
    """
    x = audio.detach().float().cpu()
    if x.dim() == 3:
        x = x.squeeze(0).squeeze(0)
    elif x.dim() == 2:
        x = x.squeeze(0)
    window = torch.hann_window(n_fft)
    spec = torch.stft(
        x,
        n_fft,
        hop_length=hop,
        return_complex=True,
        window=window,
    )
    mag = spec.abs().clamp(min=1e-12)
    freqs = torch.linspace(0, sample_rate / 2, mag.shape[0])
    centroid = (freqs[:, None] * mag).sum(dim=0) / mag.sum(dim=0)
    return centroid


def midi_pitch_contour(
    events: Sequence[tuple[int, float, float]],
    num_frames: int,
    sample_rate: int = SAMPLE_RATE,
    hop: int = 512,
) -> tuple[Tensor, Tensor]:
    """Sample the MIDI pitch contour (semitones) at the centroid frame times.

    Returns ``(contour, mask)`` of length ``num_frames``. ``mask`` is True on
    frames where a note is active; ``contour`` is the MIDI note number on
    active frames and 0 elsewhere.
    """
    times = torch.arange(num_frames) * hop / sample_rate
    contour = torch.zeros(num_frames)
    mask = torch.zeros(num_frames, dtype=torch.bool)
    for note, start, dur in sorted(events, key=lambda e: e[1]):
        lo = start
        hi = start + dur
        active = (times >= lo) & (times < hi)
        contour[active] = float(note)
        mask[active] = True
    return contour, mask


def rehydration_coherence(
    model,
    events: Sequence[tuple[int, float, float]],
    bank,
    pitch_bank_root: int = 60,
) -> dict[str, object]:
    """Phase 3b: coherence of a rehydrated render vs its MIDI score.

    Renders ``events`` with ``model.synthesize_midi`` (Mode C), computes the
    spectral-centroid contour of the render, the MIDI pitch contour sampled at
    the same frames, and reports the Pearson correlation between the two over
    frames where a note is active. A higher r means the rehydrated audio's
    spectral centroid tracks the scored pitch contour.
    """
    audio = model.synthesize_midi(
        list(events), list(bank), pitch_bank_root=pitch_bank_root, seed=3407
    )
    sr = model.sample_rate
    centroid = spectral_centroid(audio, sample_rate=sr)
    num_frames = centroid.shape[0]
    contour, mask = midi_pitch_contour(events, num_frames, sample_rate=sr)
    if mask.sum() >= 2:
        a = contour[mask].numpy().astype(np.float64)
        b = centroid[mask].numpy().astype(np.float64)
        if a.std() > 0 and b.std() > 0:
            r = float(np.corrcoef(a, b)[0, 1])
        else:
            r = 0.0
    else:
        r = float("nan")
    n_notes = int(mask.sum().item())
    return {
        "midi_events": len(events),
        "active_frames": n_notes,
        "pearson_r": r,
        "centroid_mean_hz": float(centroid.mean().item()),
        "centroid_std_hz": float(centroid.std().item()),
    }


# --------------------------------------------------------------------------- #
# Phase 3c: recursive variant diversity
# --------------------------------------------------------------------------- #


def variant_diversity(
    make_variants,
    depths: Sequence[int],
    encoder: EnCodecEncoder,
    device: torch.device,
) -> list[dict[str, object]]:
    """Mean pairwise CLAP-proxy cosine *distance* vs recursion depth.

    ``make_variants(d)`` must return a list of ``d`` audio clips (one per
    distinct MIDI at "depth" d). The function embeds them via the FAD-proxy
    audio embedding and reports the mean pairwise cosine distance
    (1 - cosine) among them; depth 1 trivially yields 0.0 (a single variant
    has no inter-variant distance).

    Returns one row per depth: ``depth, n_variants, mean_distance,
    min_distance, max_distance``.
    """
    rows: list[dict[str, object]] = []
    for d in depths:
        clips = list(make_variants(d))
        if d <= 1 or len(clips) < 2:
            rows.append(
                {
                    "depth": d,
                    "n_variants": len(clips),
                    "mean_distance": 0.0,
                    "min_distance": 0.0,
                    "max_distance": 0.0,
                }
            )
            continue
        feats = encodec_pooled_features(clips, encoder, device)
        feats = feats.to(device)
        norms = feats.norm(dim=1, keepdim=True).clamp(min=1e-12)
        unit = feats / norms
        sim = unit @ unit.T
        dist = 1.0 - sim
        iu = torch.triu_indices(d, d, offset=1)
        pairs = dist[iu[0], iu[1]]
        rows.append(
            {
                "depth": d,
                "n_variants": d,
                "mean_distance": float(pairs.mean().item()),
                "min_distance": float(pairs.min().item()),
                "max_distance": float(pairs.max().item()),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# CSV writer helper
# --------------------------------------------------------------------------- #


def write_csv(path: str | Path, rows: Sequence[dict[str, object]]) -> None:
    """Write a list of homogeneous dicts to a CSV (header from first row keys)."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
