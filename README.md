# Latent Sound Diffusion (ALD-SC)

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
modules. The research programme — decoding on the feature-space
manifold — is unchanged.

> **Central claim:** decoding on the feature-space manifold
> $(L_F, \lambda^{\mathrm{ED}})$ yields better global semantic coherence
> under compression than decoding on an unconstrained ambient latent.

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

The **baseline decoder** has identical channel widths and upsampling
strides, but uses plain `ResBlock1d` in place of `WaveReconstructionBlock`
(no $U_q$, no $\lambda^{\mathrm{ED}}$). This isolates graph structure as
the only variable.

---

## What is implemented

| Component | File | Description |
|-----------|------|-------------|
| ArrowSpace adapter | `wire_graph.py` | $L_F$ + $\lambda^{\mathrm{ED}}$ via pyarrowspace or kNN fallback |
| Frozen prior | `arrow_prior.py`, `build_prior.py` | $L_F$, $U_q$, $\Pi_q$, $c_{\mathrm{spec}}$ as buffers |
| 2.5-D encoding target | `dual_space.py` | $M_N = \alpha\|VV^\top\|_F - \beta\|V L_F V^\top\|_F$ |
| Audio codec | `audio_codec.py` | Frozen EnCodec encoder, baseline + graph decoders (planned) |
| 1-D DiT denoiser | `dit.py` | 1-D patchify + AdaLN + CFG dropout (planned) |
| Schedules | `schedule.py` | Cosine + linear, v-prediction |
| Graph decoder | `graph_decoder.py` | 1-D `WaveReconstructionBlock`, `GraphDecoder`, `ClockGatedGraphDecoder` (planned) |
| Entropic clock | `spectral_schedule.py` | $\tau_k(t)$, $\bar\alpha_k(t)$, heat-death stopping criterion |
| Samplers | `sampling.py` | DDIM + Euler with spectral stopping criterion |
| Losses | `losses.py` | $L_{\mathrm{diff}}$ + $L_{\mathrm{rec}}$ (L1+STFT) + $L_{\mathrm{chart}}$ + $L_{\mathrm{smooth}}$ |
| Training | `trainer.py` | `train_audio_decoder()` + `train_audio_diffusion()` (planned) |
| Data | `data.py` | `Esc50Dataset`, `AudioFolderDataset`, `ToyAudioDataset` (planned) |
| CLI | `scripts/sample_audio.py` | End-to-end audio generation (planned) |

---

## Status

| Phase | Scope | State |
|---|---|---|
| Setup | Project rename, audio deps, docs identity | ✅ Complete |
| Phase 1 | 1-D DiT + graph decoder + EnCodec + ESC-50 + notebook | In progress |
| Phase 2 | Music-specific generation (text/CLAP, genre, long-form) | Future |
| Phase 3 | Advanced ESDM concepts (wave recurrence, entropy clock training) | Future |

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency
management and requires Python ≥ 3.13 with PyTorch ≥ 2.2.

```bash
git clone https://github.com/tuned-org-uk/latent-sound-diffusion.git
cd latent-sound-diffusion
uv sync
```

## Usage

```bash
# Run the test suite (CPU)
uv run pytest tests/ -v

# Lint and format
uv run ruff check src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/

# Generate audio (planned)
uv run python scripts/sample_audio.py --out results/sample.wav

# With options (planned)
uv run python scripts/sample_audio.py --steps 50 --seed 3407 --out results/sample.wav
```

### Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_sound_generation.ipynb` | End-to-end sound generation with interactive knobs (planned) |

---

## Repository layout

```
latent-sound-diffusion/
├── pyproject.toml                # uv / hatchling project config
├── AGENTS.md                     # contributor guide (read this first)
├── docs/
│   ├── 00.md                     # design document — the research programme
│   ├── 01.md                     # design document — ESDM transfer
│   └── 02.md                     # design document — audio adaptation
├── notebooks/
│   └── 01_sound_generation.ipynb # end-to-end notebook (planned)
├── scripts/
│   ├── build_audio_prior.py      # build prior from EnCodec features (planned)
│   ├── train_audio_decoder.py    # decoder training (planned)
│   ├── train_audio_diffusion.py  # 1-D DiT training (planned)
│   ├── sample_audio.py           # CLI audio generation (planned)
│   └── eval_audio.py             # reconstruction + FAD eval (planned)
├── src/ald_sc/
│   ├── __init__.py
│   ├── arrow_prior.py            # ArrowSpacePrior: frozen spectral prior
│   ├── audio_codec.py            # EnCodecEncoder, BaselineAudioDecoder, AudioVAE (planned)
│   ├── build_prior.py            # build_arrow_prior() from corpus embeddings
│   ├── data.py                   # Esc50Dataset, AudioFolderDataset, ToyAudioDataset (planned)
│   ├── dit.py                    # MinimalDiT: 1-D patchify + AdaLN + CFG
│   ├── dual_space.py             # DualSpaceMatrix M_N (2.5-D encoding target)
│   ├── graph_decoder.py          # 1-D WaveReconstructionBlock, GraphDecoder,
│   │                             #   ClockGatedGraphDecoder
│   ├── losses.py                 # ALDSCLoss: diff + rec (L1+STFT) + chart + smooth
│   ├── sampling.py               # sample_euler(), sample_ddim() + spectral stopping
│   ├── schedule.py               # CosineSchedule, LinearSchedule (v-prediction)
│   ├── spectral_schedule.py      # Per-mode τ_k, ᾱ_k, heat-death criterion
│   ├── trainer.py                # train_audio_decoder(), train_audio_diffusion()
│   ├── vae.py                    # (legacy image VAE — to be removed)
│   └── wire_graph.py             # ArrowSpace adapter: L_F + λ_ED
└── tests/                        # unit tests (CPU)
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
