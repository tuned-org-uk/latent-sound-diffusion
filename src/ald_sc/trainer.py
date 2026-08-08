"""Training loops for ALD-SC audio generation.

Provides ``train_audio_decoder()`` for Phase 1 decoder training (graph or
baseline) and ``train_audio_diffusion()`` for 1-D DiT training.

Training loops yield loss dicts (mirrors the from-scratch pattern) so
notebooks can plot live curves. ``log_training()`` wraps such an
iterator and emits a structured per-epoch summary event via ``structlog``
so notebooks can follow progress without polling per-batch losses.

This module must not define model architectures (per AGENTS.md §11).
"""

from __future__ import annotations

from collections.abc import Iterator

import structlog
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ald_sc._logging import configure_logging
from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.audio_codec import BaselineAudioDecoder
from ald_sc.losses import ALDSCLoss
from ald_sc.schedule import CosineSchedule

configure_logging()

__all__ = ["train_audio_decoder", "train_audio_diffusion", "log_training"]


def train_audio_decoder(
    loader: DataLoader,
    audio_vae: nn.Module,
    prior: ArrowSpacePrior,
    loss_fn: ALDSCLoss,
    epochs: int = 10,
    lr: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    noise_std: float = 0.0,
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
    noise_std : float
        Standard deviation of Gaussian noise injected into the latent z
        before decoding (latent-space augmentation). 0.0 disables it and
        reproduces the deterministic baseline. Larger values produce
        different, non-repeatable training runs — a feature for artistic
        exploration rather than a bug.

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

    decode = audio_vae.decoder
    is_baseline = isinstance(decode, BaselineAudioDecoder)

    for epoch in range(epochs):
        for batch in loader:
            x = batch.to(device) if isinstance(batch, Tensor) else batch[0].to(device)
            optimizer.zero_grad()

            z, A, c_spec = audio_vae.encoder.encode(x, prior)

            # Latent-space augmentation: noise the latent the decoder sees.
            # c_spec stays derived from the clean A so the spectral chart
            # conditioning is preserved; only the decoder input is noised.
            if noise_std > 0:
                z = z + noise_std * torch.randn_like(z)

            x_hat = decode(z) if is_baseline else decode(z, c_spec)

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
    """Wrap a training iterator, yielding records enriched with epoch stats.

    Each yielded per-batch record is enriched with a running
    ``epoch_mean_loss`` and the count of ``epoch_steps`` seen so far in the
    current epoch. A structured ``epoch`` event is emitted via ``structlog``
    when the epoch changes (and once at the end) so callers can follow
    training progress by epoch without aggregating per-batch losses.

    Parameters
    ----------
    train_iter : Iterator[dict[str, float]]
        Iterator yielding per-batch loss dicts (must contain 'epoch' and
        'loss').
    label : str
        Label attached to each emitted structlog event (e.g. "Graph decoder").

    Yields
    ------
    dict[str, float]
        Original record enriched with 'epoch_mean_loss' and 'epoch_steps'.
    """
    log = structlog.get_logger("ald_sc.trainer")
    current_epoch: int | None = None
    epoch_sum: float = 0.0
    epoch_n: int = 0

    def _flush() -> None:
        if epoch_n == 0:
            return
        log.info(
            "epoch",
            label=label,
            epoch=current_epoch,
            mean_loss=epoch_sum / epoch_n,
            steps=epoch_n,
        )

    for record in train_iter:
        epoch = int(record["epoch"])

        if current_epoch is not None and epoch != current_epoch:
            _flush()
            epoch_sum = 0.0
            epoch_n = 0

        current_epoch = epoch
        epoch_sum += float(record["loss"])
        epoch_n += 1

        yield {
            **record,
            "epoch_mean_loss": epoch_sum / epoch_n,
            "epoch_steps": epoch_n,
        }

    _flush()
