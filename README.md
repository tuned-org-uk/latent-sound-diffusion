# LSD: Latent Sound Diffusion (forked from ALD-SC)

**ArrowSpace Latent Diffusion with Spectral Chart Conditioning** — a
spectrally conditioned latent diffusion model for **sound generation**.
A frozen ArrowSpace graph-wiring prior
$(L_F, \lambda^{\mathrm{ED}})$ defines a low-dimensional semantic
manifold. A frozen EnCodec encoder produces 1-D audio latents $z$. The
diffusion process operates on $z$; the spectral chart provides global
topology-aware conditioning. A trainable 1-D graph decoder
reconstructs waveforms by decoding on the feature-space manifold.

This repository is a sound-generation-specific fork of
[arrowspace-latent-diffusion](https://github.com/tuned-org-uk/arrowspace-latent-diffusion).
The image-generation code has been replaced with 1-D audio-native
modules. Unlike ALD-SC, LSD has no falsifiable research claim — it exists
for **sound production**: generating sounds for their own sake, and
exploring what we call *the sound of the future*.

> The frozen ArrowSpace prior $(L_F, \lambda^{\mathrm{ED}})$ is used here as a
> sound-design tool, not as the object of a claim. The goal is simply to
> generate compelling, novel audio.

### Sound production, not a research claim
LSD is a **sound-generation** variant, not the ALD-SC research artifact.
There is no falsifiable claim to defend. The objective is to let artists
**train fast on their own samples** and explore what we call *the sound of
the future*. Two knobs encode the artistic requirement that training be
*interesting*, not merely reproducible:

- **`SEED = None`** — non-repeatable training. Each run samples a fresh
  seed, so every model — and every generated sound — differs. Variation
  across runs is a feature, not a bug.
- **`NOISE_INJECT`** — latent-space noise injected into the EnCodec latent
  $z$ before decoding during decoder training. 0.0 reproduces the
  deterministic baseline; larger values raise the early loss and yield
  different, more varied models. Tune for taste.

### Quick start
```python
# Run the end-to-end notebook
uv run jupyter notebook notebooks/01_sound_generation.ipynb
```

---

## The research programme

The basic point of this research programme is to design **decoding using
three structures**, all computed from the training corpus via the
[ArrowSpace library](https://github.com/tuned-org-uk/pyarrowspace):

1. **The item-space** — the 1-D audio latent $z$ (EnCodec continuous
   features) carrying local acoustic detail.
2. **The feature-space graph Laplacian** $L_F$ — its eigenvectors $U_q$
   define the smooth semantic subspace; its eigenvalues $\nu_k$ define
   entropy exchange rates.
3. **The dispersion network** $\lambda^{\mathrm{ED}}$ — ArrowSpace's
   per-feature energy-dispersion distribution
   ([arXiv:2606.21535](https://arxiv.org/abs/2606.21535)). Not a
   diagnostic, but a constructive representation of how semantic structure
   is distributed over the feature graph.

**Decoding uses $L_F$ and $\lambda^{\mathrm{ED}}$ as constructive elements
of the decoding operator**, not merely as conditioning signals. The
`WaveReconstructionBlock` propagates information along $U_q$ directions,
gated by dispersion-derived weights — the graph-theoretic analogue of the
VAE reparameterization trick.

See [`docs/00.md`](docs/00.md) § "The research programme" and
[`AGENTS.md`](AGENTS.md) §1.1 for the full design statement.

---

## Architecture

```text
                          Frozen ArrowSpace prior
               ┌──────────────────────────────────────┐
               │ L_F, U_q, Λ_q, λ_ED  (from ArrowSpace)│
               └──────────────────────────────────────┘
                               │
  audio x ── EnCodec ──► (z, A) ──► c_spec ──► 1-D DiT ──► ẑ
  (frozen)      │         │  │                      │            │
  encoder       │         │  └─ project: A U_q U_q^T               │
                │         │                                        │
                │         └─ A = pool(z) (pooled features)        │
                │                                                  │
                │              SpectralSchedule (Barontini clock)  │
                │                     │                            ▼
                └────────────── GraphDecoder ◄── ClockGated tempo ──► x̂
                                (WaveReconstructionBlock:
                                 project → gate → lift along U_q)
```

### The 1-D audio latent

Each audio clip encodes via frozen EnCodec (24 kHz mono) to:
- **z** — 1-D latent (EnCodec pre-quantization continuous features,
  D=128, T=375 for 5s clips)
- **A = pool(z)** — pooled feature field for spectral chart extraction
- **c_spec** $\in \mathbb{R}^{3q}$ — `[ẽ, λ_chart, ν]` (conditioning vector)

c_spec is **derived from z itself** (self-consistent decoding): the DiT
generates z unconditionally, then the decoder derives c_spec from the
generated z. No separate c_spec sampling is needed.

### The graph-structured decoder

The `WaveReconstructionBlock` at each resolution:
1. Pool feature activations → $A$
2. Project to chart: $\hat{H} = A \cdot U_q$ (decode along smooth directions)
3. Gate by dispersion: $g = \sigma(W \cdot c_{\mathrm{spec}})$ (energy allocation)
4. Lift back: $A' = (\hat{H} \odot g) \cdot U_q^\top$ (reconstruct in feature space)
5. Residual conv update

The `ClockGatedGraphDecoder` modulates gate strength by
$\bar\alpha_k(t)$: early in denoising (high noise) gates are weak; late
(low noise) gates are strong.

### Controlled comparison

The **baseline decoder** (`BaselineAudioDecoder`) has identical channel
widths and upsampling strides, but uses plain `ResBlock1d` in place of
`WaveReconstructionBlock` (no $U_q$, no $\lambda^{\mathrm{ED}}$). This
isolates graph structure as the only variable.

---

## What is implemented

| Component | File | Description |
|-----------|------|-------------|
| ArrowSpace adapter | `wire_graph.py` | $L_F$ + $\lambda^{\mathrm{ED}}$ via pyarrowspace or kNN fallback |
| Frozen prior | `arrow_prior.py`, `build_prior.py` | $L_F$, $U_q$, $\Pi_q$, $c_{\mathrm{spec}}$ as buffers |
| 2.5-D encoding target | `dual_space.py` | $M_N = \alpha\|VV^\top\|_F - \beta\|V L_F V^\top\|_F$ |
| Audio codec | `audio_codec.py` | Frozen `EnCodecEncoder`, `BaselineAudioDecoder`, `AudioVAE` |
| 1-D DiT denoiser | `dit.py` | 1-D `Conv1d` patchify + AdaLN + CFG dropout, `latent_shape` attr |
| Schedules | `schedule.py` | Cosine + linear, v-prediction |
| Graph decoder | `graph_decoder.py` | 1-D `WaveReconstructionBlock`, `GraphDecoder`, `ClockGatedGraphDecoder` |
| Entropic clock | `spectral_schedule.py` | $\tau_k(t)$, $\bar\alpha_k(t)$, heat-death stopping criterion |
| Samplers | `sampling.py` | DDIM + Euler with spectral stopping, 1-D `latent_shape` noise init |
| Losses | `losses.py` | $L_{\mathrm{diff}}$ + $L_{\mathrm{rec}}$ (L1+multi-scale STFT) + $L_{\mathrm{chart}}$ + $L_{\mathrm{smooth}}$ |
| Training | `trainer.py` | `train_audio_decoder()` (with `noise_std` latent augmentation) + `train_audio_diffusion()` + `log_training()` (structlog) |
| Data | `data.py` | `Esc50Dataset`, `AudioFolderDataset`, `ToyAudioDataset`, `MusicSynthDataset` (load_audio_clip falls back to soundfile if torchaudio backend is unavailable) |
| Inference contract | `inference.py` | `LSDModel`: `generate_sound_bank` (A), `condition_on_audio` (B), `synthesize_midi` (C) — see [`docs/03.md`](docs/03.md) |
| CLI scripts | `scripts/` | `build_audio_prior`, `train_audio_decoder`, `train_audio_diffusion`, `sample_audio`, `eval_audio` |
| Configs | `configs/` | `audio_decoder.yaml`, `audio_diffusion.yaml` |
| Notebook | `notebooks/01_sound_generation.ipynb` | End-to-end pipeline with interactive knobs |

**160 unit tests**, all on CPU. `uv run pytest tests/ -v`.

---

## Status

| Phase | Scope | State |
|---|---|---|
| Setup | Project rename, audio deps, docs identity | ✅ Complete |
| Phase 1 | 1-D DiT + graph decoder + EnCodec + ESC-50 + notebook | ✅ Complete |
| Phase 2 | Music-specific generation (text/CLAP, genre, long-form) | Future |
| Phase 3 | Advanced ESDM concepts (wave recurrence, entropy clock training) | Future |

### Open issues (limitations)

- [#1](https://github.com/tuned-org-uk/latent-sound-diffusion/issues/1) — Sound generation: end-to-end audio synthesis via ALD-SC
- FAD computation: `fadtk` removed due to dependency conflicts; the evaluation pipeline (`src/ald_sc/eval.py`) computes a documented **FAD-proxy** (exact Fréchet formula on frozen EnCodec pooled features instead of VGGish) and a **CLAP-proxy** (deterministic hashing text embedding); methodology recorded in each CSV's `*_method` columns
- Real-data experiments on ESC-50 with real EnCodec (notebooks 05 and 06 use real EnCodec; notebook 01 is a CPU demo)
- [#51](https://github.com/tuned-org-uk/latent-sound-diffusion/issues/51) — graph-decoder training instability: root-caused to the wave block's time-constant `U_q` delta broadcast through GroupNorm (1/σ³ gradient knife-edge, device-noise-stream dependent); fixed with a per-time-step graph filter (1×1 convs + einsum through `U_q`), plus grad clipping and a non-finite guard in `train_audio_decoder`; CPU/MPS parity locked in by `tests/test_device_parity.py` + `scripts/repro_mps_divergence.py`

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency
management and requires Python ≥ 3.13 with PyTorch ≥ 2.2. EnCodec and
torchaudio provide audio encoding/decoding; Jupyter is included for
running notebooks.

```bash
git clone https://github.com/tuned-org-uk/latent-sound-diffusion.git
cd latent-sound-diffusion
uv sync
```

## Usage

```bash
# Run the test suite (CPU; 160 tests)
uv run pytest tests/ -v

# Lint and format
uv run ruff check src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/

# Build the ArrowSpace prior from EnCodec features
uv run python scripts/build_audio_prior.py --toy --out prior.pt

# Train a decoder (graph or baseline)
uv run python scripts/train_audio_decoder.py --prior prior.pt --graph --epochs 50
uv run python scripts/train_audio_decoder.py --prior prior.pt --baseline --epochs 50

# Train the 1-D DiT
uv run python scripts/train_audio_diffusion.py --prior prior.pt --toy --epochs 50

# Generate audio
uv run python scripts/sample_audio.py --prior prior.pt --decoder decoder.pt --dit dit.pt --out results/sample.wav

# Evaluate reconstruction (graph vs baseline, λ_ED ablation)
uv run python scripts/eval_audio.py --graph-decoder decoder.pt --baseline-decoder baseline.pt --toy

# Run the end-to-end notebook
uv run jupyter notebook notebooks/01_sound_generation.ipynb
```

### Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_sound_generation.ipynb` | End-to-end sound generation with interactive knobs (`SEED`, `STEPS`, `USE_C_SPEC`, `TEMPERATURE`, `CLIP_INDEX`) |
| 02 | `02_long_form_generation.ipynb` | Long-form (5-8s) generation via overlap-and-add with pitch envelope, ADSR, and timbral filtering knobs |
| 03 | `03_bps_modulation_effects.ipynb` | BPS-synchronized modulation (tremolo, vibrato, filter sweep) with pedalboard effects and scipy.signal IIR filters |
| 04 | `04_latent_concatenation.ipynb` | Latent-space concatenation of multiple variations into ~20s output, with per-section BPS modulation and pedalboard effects |
| 05 | `05_full_music_generation.ipynb` | Full music generation with the real frozen EnCodec encoder and synthetic `MusicSynthDataset` |
| 06 | `06_end_to_end_real_data.ipynb` | End-to-end training on real audio files in `data/` using `AudioFolderDataset` and real EnCodec |
| 07 | `07_production_workflow.ipynb` | Full LSD production workflow — train two models, A/B/C inference, then longform + BPS/effects + latent-concat progression |

To run a notebook server:

```bash
# Launch Jupyter Lab (preferred)
uv run jupyter lab notebooks/

# Or classic notebook interface
uv run jupyter notebook notebooks/01_sound_generation.ipynb

# Or execute non-interactively (outputs saved in-place)
uv run jupyter nbconvert --to notebook --execute notebooks/01_sound_generation.ipynb
```

Notebook 01 runs entirely on CPU using a stub encoder and
`ToyAudioDataset` (no external data or GPU required). Notebooks 05 and 06 use
the real frozen `EnCodecEncoder`: notebook 05 with synthetic `MusicSynthDataset`,
and notebook 06 with real audio files in `data/` via `AudioFolderDataset`.

---

## Repository layout

```
latent-sound-diffusion/
├── pyproject.toml                # uv / hatchling project config
├── AGENTS.md                     # contributor guide (read this first)
├── configs/
│   ├── audio_decoder.yaml        # decoder training config
│   └── audio_diffusion.yaml      # diffusion training config
├── docs/
│   ├── 00.md                     # design document — the research programme
│   ├── 01.md                     # design document — ESDM transfer
│   └── 02.md                     # design document — audio adaptation
├── notebooks/
│   ├── 01_sound_generation.ipynb       # end-to-end notebook (stub encoder)
│   ├── 02_long_form_generation.ipynb   # overlap-and-add + effects
│   ├── 03_bps_modulation_effects.ipynb  # BPS modulation + pedalboard
│   ├── 04_latent_concatenation.ipynb   # latent-space concatenation
│   ├── 05_full_music_generation.ipynb  # real EnCodec encoder
│   ├── 06_end_to_end_real_data.ipynb   # train on audio files in data/
│   └── 07_production_workflow.ipynb  # full production workflow (A/B/C + effects chain)
├── scripts/
│   ├── build_audio_prior.py      # build prior from EnCodec features
│   ├── train_audio_decoder.py    # decoder training (graph vs baseline)
│   ├── train_audio_diffusion.py  # 1-D DiT training
│   ├── sample_audio.py           # CLI audio generation
│   └── eval_audio.py             # reconstruction + FAD eval
├── src/ald_sc/
│   ├── __init__.py
│   ├── arrow_prior.py            # ArrowSpacePrior: frozen spectral prior
│   ├── audio_codec.py            # EnCodecEncoder, BaselineAudioDecoder, AudioVAE
│   ├── build_prior.py            # build_arrow_prior() from corpus embeddings
│   ├── data.py                   # Esc50Dataset, AudioFolderDataset, ToyAudioDataset
│   ├── dit.py                    # MinimalDiT: 1-D patchify + AdaLN + CFG
│   ├── dual_space.py             # DualSpaceMatrix M_N (2.5-D encoding target)
│   ├── graph_decoder.py          # 1-D WaveReconstructionBlock, GraphDecoder,
│   │                             #   ClockGatedGraphDecoder
│   ├── losses.py                 # ALDSCLoss: diff + rec (L1+STFT) + chart + smooth
│   ├── sampling.py               # sample_euler(), sample_ddim() + spectral stopping
│   ├── schedule.py               # CosineSchedule, LinearSchedule (v-prediction)
│   ├── spectral_schedule.py      # Per-mode τ_k, ᾱ_k, heat-death criterion
│   ├── trainer.py                # train_audio_decoder(), train_audio_diffusion()
│   └── wire_graph.py             # ArrowSpace adapter: L_F + λ_ED
└── tests/                        # 16 test files, 160 tests (CPU)
```

---

## Design constraints

- **Frozen prior.** $L_F$, $U_q$ are buffers, never parameters.
- **Frozen encoder.** EnCodec weights are never updated. Only the
  decoder and DiT are trained.
- **Decoding on the feature-space manifold.** $L_F$ defines reconstruction
  paths (via $U_q$); $\lambda^{\mathrm{ED}}$ defines energy allocation.
- **Diffusion runs on $z$ only.** No second diffusion process over the
  spectral chart $s$.
- **Corpus-level prior, not per-clip.** Do not construct a new graph per
  audio clip.
- **Unconditional generation (Phase 1).** No text/genre/tag conditioning.
  Text/CLAP conditioning is tracked separately for the music-generation
  follow-up.

See [`AGENTS.md`](AGENTS.md) §6 for the full design constraints.

---

## References

- Design documents: [`docs/00.md`](docs/00.md), [`docs/01.md`](docs/01.md), [`docs/02.md`](docs/02.md)
- Upstream: [`arrowspace-latent-diffusion`](https://github.com/tuned-org-uk/arrowspace-latent-diffusion)
- [Diffusion as spectral-geometric projection](https://www.tuned.org.uk/posts/021_diffusion_as_spectral_geometric_projection/)
- [`entropic-semantic-diffusion`](https://github.com/tuned-org-uk/entropic-semantic-diffusion)
- [`pyarrowspace`](https://github.com/tuned-org-uk/pyarrowspace) — ArrowSpace library (Rust bindings)
- [Energy Dispersion Networks](https://arxiv.org/abs/2606.21535)
- EnCodec: Défossez et al., 2022
- ESC-50: Piczak, 2015
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (CVPR 2022)
- Barontini, *Testing the problem of time with cold atoms* (PRL 2026)
- Stancevic et al., *Entropic Time Schedulers for Generative Diffusion Models* (arXiv 2025)

## License

MIT
