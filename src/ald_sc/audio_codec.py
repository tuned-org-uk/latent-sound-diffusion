"""Audio codec: frozen EnCodec encoder + trainable graph/baseline decoders.

EnCodec (24 kHz mono) is used as a **frozen feature extractor**. Its
pre-quantization continuous encoder features serve as the 1-D diffusion
latent z. We train a new 1-D decoder (graph or baseline) to map z back
to a waveform.

The graph decoder uses L_F (via U_q) for reconstruction paths and λ_ED
for energy allocation — the research contribution. The baseline decoder
has matched capacity but no graph structure, isolating graph structure
as the only variable in the controlled comparison.

This module must not add training logic (per AGENTS.md §11).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ald_sc.arrow_prior import ArrowSpacePrior

if TYPE_CHECKING:
    from encodec import EncodecModel

__all__ = [
    "EnCodecEncoder",
    "BaselineAudioDecoder",
    "AudioVAE",
    "extract_encodec_features",
]

logger = logging.getLogger(__name__)


def _migrate_weight_norm(model: nn.Module) -> int:
    """Convert deprecated ``weight_norm`` hooks to the new parametrization API.

    PyTorch deprecated ``torch.nn.utils.weight_norm`` (a forward pre-hook that
    materialises ``weight_g``/``weight_v``) in favour of
    ``torch.nn.utils.parametrizations.weight_norm`` (a proper parametrization).
    The third-party ``encodec`` package still constructs its conv stack with
    the old hook, which emits a ``FutureWarning`` on every load and will raise
    a hard error once PyTorch removes the old API.

    This iterates every module carrying the deprecated ``WeightNorm`` forward
    pre-hook, removes it (merging ``weight_g``/``weight_v`` back into
    ``weight``) and re-applies the new parametrization with the same ``name``
    and ``dim``. The effective weights are preserved (the round-trip is the
    identity up to floating point), so model outputs are unchanged.

    Parameters
    ----------
    model : nn.Module
        Model whose modules may carry the deprecated hook.

    Returns
    -------
    int
        Number of modules migrated (``0`` means nothing to do).
    """
    from torch.nn.utils.weight_norm import WeightNorm

    migrated = 0
    for module in model.modules():
        hooks = [
            h for h in module._forward_pre_hooks.values() if isinstance(h, WeightNorm)
        ]
        for hook in hooks:
            name = hook.name
            dim = hook.dim
            torch.nn.utils.remove_weight_norm(module, name)
            torch.nn.utils.parametrizations.weight_norm(module, name=name, dim=dim)
            migrated += 1
    return migrated


class EnCodecEncoder(nn.Module):
    """Frozen EnCodec encoder wrapper for audio latent extraction.

    Uses the 24 kHz mono EnCodec model's encoder to produce continuous
    pre-quantization features z ∈ (B, 128, T). These serve as the 1-D
    diffusion latent space.

    The model is loaded once and frozen (zero trainable parameters).

    Parameters
    ----------
    sample_rate : int
        Target sample rate (default 24000 for EnCodec 24 kHz).
    bandwidth : int
        Target bandwidth in kbps (affects number of codebooks, but we
        use pre-quantization features regardless). Default 24.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        bandwidth: int = 24,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self._encodec: EncodecModel | None = None
        self._loaded = False
        self._device = torch.device("cpu")

    def to(self, *args, **kwargs):
        """Track the target device so lazy-loaded EnCodec lands on it."""
        if args:
            self._device = torch.device(args[0])
        elif "device" in kwargs:
            self._device = torch.device(kwargs["device"])
        if self._loaded and self._encodec is not None:
            self._encodec = self._encodec.to(self._device)
        return super().to(*args, **kwargs)

    def _load_model(self) -> None:
        """Lazily load the EnCodec model (deferred to avoid import on init).

        The third-party ``encodec`` package builds its conv stack with the
        deprecated ``torch.nn.utils.weight_norm`` hook, which raises a
        ``FutureWarning`` on construction and will become a hard error once
        PyTorch removes that API (see issue #33). We suppress that
        construction-time warning and immediately migrate every affected
        module to the new ``torch.nn.utils.parametrizations.weight_norm``
        parametrization, so the effective weights are preserved but the
        model no longer depends on the deprecated hook at inference time.
        """
        if self._loaded:
            return
        try:
            import warnings

            from encodec import EncodecModel

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*weight_norm.*deprecated.*",
                    category=FutureWarning,
                )
                self._encodec = EncodecModel.encodec_model_24khz()
                self._encodec.set_target_bandwidth(self.bandwidth)
                migrated = _migrate_weight_norm(self._encodec)
            if migrated:
                logger.debug(
                    "migrated %d weight_norm hook(s) to parametrization API",
                    migrated,
                )
            for p in self._encodec.parameters():
                p.requires_grad_(False)
            self._encodec.eval()
            self._encodec = self._encodec.to(self._device)
            self._loaded = True
        except Exception as e:
            logger.warning("Failed to load EnCodec model: %s", e)
            raise

    def encode(
        self, x: Tensor, prior: ArrowSpacePrior
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode audio waveform to (z, A, c_spec).

        Parameters
        ----------
        x : Tensor (B, 1, T_audio)
            Raw audio waveform at 24 kHz mono.
        prior : ArrowSpacePrior
            Frozen prior for spectral chart extraction.

        Returns
        -------
        z : Tensor (B, 128, T_frames)
            EnCodec continuous pre-quantization latent.
        A : Tensor (B, 128)
            Pooled feature field (mean over temporal axis).
        c_spec : Tensor (B, 3*q)
            Spectral conditioning vector.
        """
        self._load_model()
        assert self._encodec is not None

        with torch.no_grad():
            # EnCodec encoder: (B, 1, T_audio) -> (B, 128, T_frames)
            z = self._encodec.encoder(x)
            # Ensure float32 (EnCodec may use float16)
            z = z.float()

        # Pool over temporal axis: (B, 128, T) -> (B, 128)
        a = z.mean(dim=2)

        # Spectral chart conditioning: (B, 128) -> (B, 3*q)
        c_spec = prior.chart_energy_descriptor(a)

        return z, a, c_spec

    def extract_features(self, x: Tensor) -> Tensor:
        """Extract continuous EnCodec features (no prior needed).

        Parameters
        ----------
        x : Tensor (B, 1, T_audio)
            Raw audio waveform.

        Returns
        -------
        Tensor (B, 128, T_frames)
            Continuous encoder features.
        """
        self._load_model()
        assert self._encodec is not None
        with torch.no_grad():
            z = self._encodec.encoder(x)
            return z.float()


class ResBlock1d(nn.Module):
    """Simple 1-D residual block with GroupNorm and SiLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = nn.functional.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.skip(x) + h


class BaselineAudioDecoder(nn.Module):
    """Matched-capacity 1-D conv decoder (no graph structure).

    Same channel widths and upsampling strides as GraphDecoder, but
    uses plain ResBlock1d in place of WaveReconstructionBlock — no U_q,
    no λ_ED gating. This is the unconstrained baseline for the controlled
    comparison.

    Parameters
    ----------
    latent_channels : int
        Latent channels (128 for EnCodec).
    out_channels : int
        Output audio channels (1 for mono).
    base_channels : int
        Base width for conv layers.
    upsample_strides : tuple[int, ...]
        Upsampling factors per stage (default (2,4,5,8) for 320×).
    """

    def __init__(
        self,
        latent_channels: int = 128,
        out_channels: int = 1,
        base_channels: int = 64,
        upsample_strides: tuple[int, ...] = (2, 4, 5, 8),
    ) -> None:
        super().__init__()
        self.upsample_strides = upsample_strides

        ch = base_channels

        self.dec_in = nn.Conv1d(latent_channels, ch * 4, 1)

        self.res_blocks = nn.ModuleList()
        self.dec_stages = nn.ModuleList()

        channel_steps = [ch * 4, ch * 2, ch, ch, ch]
        while len(channel_steps) < len(upsample_strides) + 1:
            channel_steps.append(ch)

        for i, stride in enumerate(upsample_strides):
            in_ch = channel_steps[i]
            out_ch = channel_steps[i + 1]
            self.res_blocks.append(ResBlock1d(in_ch, in_ch))
            self.dec_stages.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, 3, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                )
            )

        self.dec_out = nn.Conv1d(
            channel_steps[len(upsample_strides)], out_channels, 3, padding=1
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decode 1-D latent to waveform (no spectral conditioning).

        Parameters
        ----------
        z : Tensor (B, latent_channels, T)

        Returns
        -------
        Tensor (B, out_channels, T * prod(upsample_strides))
        """
        h = self.dec_in(z)

        for i, stride in enumerate(self.upsample_strides):
            h = self.res_blocks[i](h)
            h = self.dec_stages[i](h)
            h = nn.functional.interpolate(h, scale_factor=stride, mode="nearest")

        x_hat = self.dec_out(h)
        return x_hat


class AudioVAE(nn.Module):
    """Audio VAE: frozen EnCodec encoder + trainable decoder.

    Composes a frozen EnCodec encoder with either a graph-structured
    decoder (GraphDecoder / ClockGatedGraphDecoder) or a baseline
    decoder (BaselineAudioDecoder).

    Parameters
    ----------
    encoder : EnCodecEncoder
        Frozen EnCodec encoder.
    decoder : nn.Module
        Trainable decoder (GraphDecoder, ClockGatedGraphDecoder, or
        BaselineAudioDecoder).
    """

    # Declared submodule types (nn.Module.__getattr__ would otherwise widen
    # attribute access to Tensor | Module for static checkers).
    encoder: "EnCodecEncoder"
    decoder: nn.Module

    def __init__(self, encoder: EnCodecEncoder, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, x: Tensor, prior: ArrowSpacePrior
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full encode-decode pass.

        Parameters
        ----------
        x : Tensor (B, 1, T_audio)
            Raw audio waveform.
        prior : ArrowSpacePrior
            Frozen prior for spectral chart extraction.

        Returns
        -------
        z : Tensor (B, 128, T_frames)
        A : Tensor (B, 128)
        c_spec : Tensor (B, 3*q)
        x_hat : Tensor (B, out_channels, T_audio)
        """
        z, a, c_spec = self.encoder.encode(x, prior)

        # Decode: graph decoder takes (z, c_spec), baseline takes z only
        if isinstance(self.decoder, BaselineAudioDecoder):
            x_hat = self.decoder(z)
        else:
            x_hat = self.decoder(z, c_spec)

        return z, a, c_spec, x_hat


def extract_encodec_features(
    loader: torch.utils.data.DataLoader,
    encoder: EnCodecEncoder,
) -> Tensor:
    """Extract pooled EnCodec features from a dataloader for prior building.

    Iterates over all clips in the dataloader, encodes each with the
    frozen EnCodec encoder, pools over the temporal axis, and stacks
    into a (N, F) corpus embedding matrix.

    Parameters
    ----------
    loader : DataLoader
        Audio dataloader yielding (B, 1, T) waveforms.
    encoder : EnCodecEncoder
        Frozen EnCodec encoder.

    Returns
    -------
    Tensor (N, F)
        Corpus embeddings where F=128 (EnCodec encoder dim).
    """
    features: list[Tensor] = []
    for batch in loader:
        x = batch if isinstance(batch, Tensor) else batch[0]
        z = encoder.extract_features(x)
        a = z.mean(dim=2)  # (B, 128)
        features.append(a)
    return torch.cat(features, dim=0)
