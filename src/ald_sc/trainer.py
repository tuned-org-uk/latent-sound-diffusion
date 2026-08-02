"""Training loops for ALD-SC audio generation.

Provides ``train_audio_decoder()`` for Phase 1 decoder training (graph or
baseline) and ``train_audio_diffusion()`` for 1-D DiT training.

Training loops yield loss dicts (mirrors the from-scratch pattern) so
notebooks can plot live curves.

This module must not define model architectures (per AGENTS.md §11).
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.losses import ALDSCLoss
from ald_sc.schedule import CosineSchedule

__all__ = ["train_audio_decoder", "train_audio_diffusion", "log_training"]


def train_audio_decoder(
    loader: DataLoader,
    audio_vae: nn.Module,
    prior: ArrowSpacePrior,
    loss_fn: ALDSCLoss,
    epochs: int = 10,
    lr: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> Iterator[dict[str, float]]:
    """Phase 1 audio decoder training loop.

    Freezes the EnCodec encoder and trains the decoder (graph or baseline)
    to reconstruct waveforms from EnCodec latents.

    Parameters
    ----------
    loader : DataLoader
        Audio dataloader yielding (B, 1, T) waveforms.
    audio_vae : nn.Module
        AudioVAE (encoder + decoder). Encoder is frozen, decoder is trained.
    prior : ArrowSpacePrior
        Frozen ArrowSpace prior.
    loss_fn : ALDSCLoss
        Loss function instance.
    epochs : int
    lr : float
    device : torch.device

    Yields
    ------
    dict[str, float]
        Loss dict with 'epoch', 'loss', 'rec', 'stft', 'chart', 'smooth'.
    """
    audio_vae = audio_vae.to(device)
    prior = prior.to(device)

    # Freeze encoder
    for p in audio_vae.encoder.parameters():
        p.requires_grad_(False)

    # Only train decoder
    decoder_params = [p for p in audio_vae.decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(decoder_params, lr=lr)

    for epoch in range(epochs):
        for batch in loader:
            x = batch.to(device) if isinstance(batch, Tensor) else batch[0].to(device)
            optimizer.zero_grad()

            z, A, c_spec, x_hat = audio_vae(x, prior)

            A_hat = A.detach()
            losses = loss_fn(x, x_hat, A, A_hat)

            losses["total"].backward()
            optimizer.step()

            yield {
                "epoch": epoch,
                "loss": float(losses["total"].item()),
                "rec": float(losses["rec"].item()),
                "stft": float(losses["stft"].item()),
                "chart": float(losses["chart"].item()),
                "smooth": float(losses["smooth"].item()),
            }


def train_audio_diffusion(
    loader: DataLoader,
    audio_vae: nn.Module,
    dit: nn.Module,
    prior: ArrowSpacePrior,
    schedule: CosineSchedule,
    epochs: int = 10,
    lr: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    cfg_dropout: float = 0.1,
) -> Iterator[dict[str, float]]:
    """Phase 1 audio diffusion training loop.

    The VAE (encoder + trained decoder) and prior are frozen; only the
    1-D DiT is trained. The DiT operates **unconditionally** (time-only
    AdaLN) — c_spec is used by the decoder, not the DiT.

    Parameters
    ----------
    loader : DataLoader
        Audio dataloader yielding (B, 1, T) waveforms.
    audio_vae : nn.Module
        Frozen AudioVAE (encoder used for z0 extraction).
    dit : nn.Module
        1-D DiT denoiser to train.
    prior : ArrowSpacePrior
        Frozen prior.
    schedule : CosineSchedule
        Noise schedule.
    epochs : int
    lr : float
    device : torch.device
    cfg_dropout : float
        Probability of dropping c_spec (unused in unconditional Phase 1,
        kept for future text/CLAP conditioning).

    Yields
    ------
    dict[str, float]
        Loss dict with 'epoch', 'loss'.
    """
    audio_vae = audio_vae.to(device).eval()
    prior = prior.to(device)
    dit = dit.to(device)

    for p in audio_vae.parameters():
        p.requires_grad_(False)
    for p in prior.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(dit.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in loader:
            x = batch.to(device) if isinstance(batch, Tensor) else batch[0].to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                z0, A, c_spec = audio_vae.encoder.encode(x, prior)

            t = schedule.sample_batch(z0)
            noise = torch.randn_like(z0)
            z_t = schedule.add_noise(z0, t, noise)
            v_target = schedule.v_target(z0, t, noise)

            # Unconditional: DiT uses time-only AdaLN (no c_spec)
            v_pred = dit(z_t, t)

            loss = (v_pred - v_target).pow(2).mean()
            loss.backward()
            optimizer.step()

            yield {
                "epoch": epoch,
                "loss": float(loss.item()),
            }


def log_training(
    train_iter: Iterator[dict[str, float]],
    label: str = "Training",
) -> Iterator[dict[str, float]]:
    """Consume a training iterator and yield epoch-level summaries.

    Each yielded record is enriched with ``epoch_mean_loss`` and
    ``epoch_steps``. A per-epoch summary line is printed whenever the
    epoch changes. This lets notebooks follow training progress without
    manually aggregating per-batch loss dicts.

    Parameters
    ----------
    train_iter : Iterator[dict[str, float]]
        Iterator yielding per-batch loss dicts (must contain 'epoch' and
        'loss').
    label : str
        Label printed in summary lines (e.g. "Graph decoder").

    Yields
    ------
    dict[str, float]
        Original record enriched with 'epoch_mean_loss' and 'epoch_steps'.
    """
    current_epoch: int | None = None
    epoch_records: list[dict[str, float]] = []

    def _flush() -> None:
        if not epoch_records:
            return
        mean_loss = sum(r["loss"] for r in epoch_records) / len(epoch_records)
        epoch = epoch_records[0]["epoch"]
        print(
            f"{label} epoch {epoch}: loss={mean_loss:.4f} ({len(epoch_records)} steps)"
        )

    for record in train_iter:
        epoch = int(record["epoch"])

        if current_epoch is not None and epoch != current_epoch:
            _flush()
            epoch_records = []

        current_epoch = epoch
        epoch_records.append(record)

        mean_loss = sum(r["loss"] for r in epoch_records) / len(epoch_records)
        enriched = {
            **record,
            "epoch_mean_loss": float(mean_loss),
            "epoch_steps": len(epoch_records),
        }
        yield enriched

    _flush()
