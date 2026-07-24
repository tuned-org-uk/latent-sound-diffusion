# latent-sound-diffusion

Name: Latent Sound Diffusion

## 1. Project Identity

**ALD-SC** (ArrowSpace Latent Diffusion with Spectral Chart Conditioning) is a
spectrally conditioned latent diffusion model for **sound generation**. A
frozen ArrowSpace graph-wiring prior (L_F, U_q, Λ_q, λ_ED) defines a
low-dimensional semantic manifold. A frozen EnCodec encoder produces 1-D
audio latents `z` carrying local acoustic detail. The diffusion process
operates only on the 1-D latent; the spectral chart provides global
topology-aware conditioning.

This repository is a sound-generation-specific fork of
[arrowspace-latent-diffusion](https://github.com/tuned-org-uk/arrowspace-latent-diffusion).
The image-generation code has been replaced with 1-D audio-native
modules. The research programme is unchanged (see §1.1).

The architecture is deliberately simple: it reuses ESDM's frozen-prior
principle without replicating its full wave-recurrence and entropy-clock
machinery. It reuses concepts developed in
https://github.com/tuned-org-uk/entropic-semantic-diffusion

### 1.1 The research programme: decoding on the feature-space manifold

The basic point of this research programme is to design **decoding using
three structures**, all computed from the training corpus via the
ArrowSpace library (https://github.com/tuned-org-uk/pyarrowspace):

1. **The item-space** — the 1-D audio latent `z` (EnCodec continuous
   features) carrying local acoustic detail.
2. **The feature-space graph Laplacian** L_F — its eigenvectors U_q define
   the smooth semantic subspace; its eigenvalues ν_k define entropy
   exchange rates.
3. **The dispersion network** λ_ED — ArrowSpace's per-feature
   energy-dispersion distribution (https://arxiv.org/abs/2606.21535).
   This is not a diagnostic; it is a representation of how semantic
   structure is distributed over the feature graph.

**Decoding must use L_F and λ_ED as constructive elements of the
decoding operator, not merely as conditioning signals.** L_F defines the
reconstruction paths (which directions to reconstruct along);
λ_ED defines the energy allocation (where to concentrate reconstruction
effort). Standard VAEs decode via the reparameterization trick (a
continuous surface integral over a Gaussian latent). The research target
is to find an analogous construction using the graph structures.

See `docs/00.md` § "The research programme" for the full design
statement. The central falsifiable claim is that decoding on the
feature-space manifold yields better global semantic coherence under
compression than decoding on an unconstrained ambient latent.

***

## 2. Repository Layout

```
latent-sound-diffusion/
├── pyproject.toml              # uv / hatchling project config
├── README.md
├── AGENTS.md                   # this file
├── .gitignore
├── configs/                    # YAML configs (planned)
│   ├── audio_decoder.yaml      #   decoder training config
│   └── audio_diffusion.yaml    #   diffusion training config
├── docs/
│   ├── 00.md                   # design document — the research programme
│   ├── 01.md                   # design document — ESDM transfer
│   └── 02.md                   # design document — audio adaptation
├── notebooks/                  # numbered milestones
│   └── 01_sound_generation.ipynb  # end-to-end notebook (planned)
├── scripts/
│   ├── build_audio_prior.py    # build frozen prior from EnCodec features
│   ├── train_audio_decoder.py  # decoder training (graph vs baseline)
│   ├── train_audio_diffusion.py # 1-D DiT training
│   ├── sample_audio.py        # inference / audio generation
│   └── eval_audio.py           # reconstruction + FAD evaluation
├── src/ald_sc/
│   ├── __init__.py
│   ├── arrow_prior.py          # ArrowSpacePrior: frozen spectral prior
│   ├── audio_codec.py          # EnCodecEncoder, BaselineAudioDecoder, AudioVAE
│   ├── build_prior.py          # build_arrow_prior() from corpus embeddings
│   ├── data.py                 # Esc50Dataset, AudioFolderDataset, ToyAudioDataset
│   ├── dit.py                  # MinimalDiT: 1-D patchify + AdaLN transformer
│   ├── dual_space.py           # DualSpaceMatrix M_N (2.5-D encoding target)
│   ├── graph_decoder.py        # WaveReconstructionBlock, GraphDecoder,
│   │                           #   ClockGatedGraphDecoder (1-D audio)
│   ├── losses.py               # ALDSCLoss: rec (L1+STFT) + chart + smooth
│   ├── sampling.py             # sample_euler(), sample_ddim() + spectral stopping
│   ├── schedule.py             # CosineSchedule, LinearSchedule (v-prediction)
│   ├── spectral_schedule.py    # Per-mode τ_k, ᾱ_k, heat-death criterion
│   ├── trainer.py              # train_audio_decoder(), train_audio_diffusion()
│   └── wire_graph.py           # ArrowSpace adapter: L_F + λ_ED
└── tests/                      # unit tests (CPU)
```

***

## 3. Environment Setup

This project uses **uv** for dependency management.

```bash
# Clone and install
git clone https://github.com/tuned-org-uk/latent-sound-diffusion.git
cd latent-sound-diffusion
uv sync

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/ scripts/
```

Always use the local latent-sound-diffusion/.venv environment.

Python ≥ 3.13 is required. PyTorch ≥ 2.2 is the primary dependency.
EnCodec and torchaudio provide audio encoding/decoding; fadtk provides
Fréchet Audio Distance computation.

***

## 4. Core Concepts (Quick Reference)

### 4.1 The Frozen ArrowSpace Prior

The prior is built **once** from a corpus of feature embeddings and never
updated during training. It consists of:

| Symbol | Shape | Description |
|---|---|---|
| L_F | (F, F) | Feature-space graph Laplacian |
| U_q | (F, q) | Leading q eigenvectors (smooth modes) |
| eigvals_q (ν) | (q,) | Corresponding eigenvalues |
| lambdas_ed (λ_ED) | (F,) | Energy-dispersion distribution per feature node |
| lambdas_chart | (q,) | λ_ED projected onto chart: Σ_f λ_f U²_{f,k} |

Key methods on `ArrowSpacePrior`:

```python
prior.project_to_chart(A)       # A @ U_q @ U_q^T  — smooth projection
prior.off_manifold_energy(A)    # relative Frobenius energy outside smooth subspace
prior.chart_coefficients(A)     # s = Pool(A) @ U_q  — spectral chart coords
prior.band_energies(A)          # e_k = ||A u_k||^2 / N  — per-mode energy
prior.chart_energy_descriptor(A) # [ẽ, λ_chart, ν]  — conditioning vector (3*q,)
```

### 4.2 The 1-D Audio Latent

Each audio clip encodes via frozen EnCodec to:

- **z ∈ R^{B×D×T}** — 1-D latent (EnCodec pre-quantization continuous
  features, D=128, T=375 for 5s @ 24kHz)
- **A = pool(z) ∈ R^{B×D}** — pooled feature field (D=F=128)

The ArrowSpace projection restricts A to the smooth subspace. The
spectral chart provides compact global conditioning via `c_spec`.

### 4.3 Noise Schedule

- `CosineSchedule` — ScheduleLDM-style, for diffusion in latent space
- `LinearSchedule` — ScheduleDDPM-style, for comparison

Both support `add_noise(z0, t, noise)` and `v_target(z0, t, noise)` for
v-prediction training.

### 4.4 Loss Components

| Loss | Formula | When active |
|---|---|---|
| L_diff | \|\|v − v_θ(z_t, t, c_spec)\|\|² | Diffusion training |
| L_rec | \|\|x − x̂\|\|₁ + λ_stft·L_STFT | Decoder training |
| L_chart | \|\|ẽ(x) − ẽ(x̂)\|\|² | Decoder training |
| L_smooth | \|\|A(I − U_qU_q^T)\|\|²_F / \|\|A\|\|²_F | Decoder training |

***

## 5. Development Workflow

### 5.1 Standard Development Cycle

1. **Understand the design**: Read `docs/00.md` and `docs/02.md` first.
2. **Run tests**: `uv run pytest tests/ -v` — all must pass before pushing.
3. **Lint**: `uv run ruff check src/ tests/ scripts/` — fix all warnings.
4. **Format**: `uv run ruff format src/ tests/ scripts/`
5. **Commit**: Use conventional commit messages (see §8).

### 5.2 Adding a New Module

1. Create the file in `src/ald_sc/`.
2. Export public symbols in `src/ald_sc/__init__.py`.
3. Add corresponding test file in `tests/`.
4. Update `docs/02.md` if the architecture changes.
5. Run `uv run pytest tests/ -v` and `uv run ruff check`.

### 5.3 Modifying an Existing Module

1. Read the module docstring and existing tests first.
2. Make the minimal change needed.
3. Update or add tests to cover the change.
4. Run tests and lint.
5. If the change affects the architecture, update `docs/02.md`.

***

## 6. Architecture Decisions and Constraints

### 6.1 What ALD-SC Is

- A latent diffusion model for audio with an additional frozen
  spectral conditioning path derived from ArrowSpace graph wiring.
- The diffusion backbone is a minimal 1-D DiT; swap for a larger
  architecture when scaling up.
- EnCodec (frozen) provides the encoder. The decoder is a 1-D
  graph-structured decoder with spectral-chart gating, trained against
  a matched-capacity unconstrained baseline.

### 6.2 What ALD-SC Is Not

- **Not a full ESDM.** The wave recurrence, entropy clock, Rayleigh-gradient
  restoring force, and learned pump are intentionally deferred.
- **Not a per-clip graph.** The ArrowSpace prior is corpus-level and frozen.
- **Not a coupled dual-diffusion.** Diffusion runs on z only.
- **Not text/genre-conditioned (Phase 1).** Unconditional generation only;
  text/CLAP conditioning is tracked in a follow-up issue.

### 6.3 Design Principles

1. **Frozen prior**: L_F, U_q are buffers, not Parameters.
2. **Frozen encoder**: EnCodec weights are never updated.
3. **Simple and clean**: Keep the architecture readable.
4. **Controlled comparison**: Graph decoder vs baseline decoder at
   matched capacity — the only variable is graph structure.

***

## 7. Implementation Phases

### Phase 1: Sound generation (current focus)

**Goal**: End-to-end audio synthesis via ALD-SC. Validate that
graph-structured decoding improves reconstruction fidelity over an
unconstrained baseline for audio.

- [ ] `dit.py` — 1-D DiT with Conv1d patchify + AdaLN (Issue #3)
- [ ] `graph_decoder.py` — 1-D WaveReconstructionBlock + GraphDecoder (Issue #4)
- [ ] `audio_codec.py` — EnCodecEncoder + BaselineAudioDecoder + AudioVAE (Issue #5)
- [ ] `data.py` — Esc50Dataset, AudioFolderDataset, ToyAudioDataset (Issue #5)
- [ ] `sampling.py` — 1-D latent sampling (Issue #6)
- [ ] `losses.py` — L1 + multi-scale STFT + chart + smooth (Issue #6)
- [ ] `trainer.py` — train_audio_decoder() + train_audio_diffusion() (Issue #6)
- [ ] Scripts + configs (Issue #7)
- [ ] End-to-end notebook with interactive knobs (Issue #8)

### Phase 2: Music-specific generation (future)

- Text/CLAP conditioning
- Genre/instrument conditioning
- Long-form musical structure
- Multi-instrument / polyphonic corpora

### Phase 3: Advanced ESDM Concepts (future)

- [ ] Add entropy-clock-gated schedule
- [ ] Add wave-recurrence encoder block
- [ ] Add Rayleigh-gradient regularizer
- [ ] Explore coupled z + s diffusion

***

## 8. Commit Message Conventions

Use conventional commits:

```
feat: add 1-D DiT patchify for audio latents
fix: correct chart_energy_descriptor padding when q < F
refactor: move reparametrize to static method
test: add off_manifold_energy boundary tests
docs: update 02.md with audio decoder details
chore: bump torchaudio to 0.28
```

***

## 9. Testing Requirements

- Every public function and class must have at least one test.
- Tests must pass on CPU (no GPU required).
- Use small tensor sizes in tests (e.g., F=32, q=8, T=16).
- Test both forward pass shapes and gradient flow.
- EnCodec-dependent tests must skip gracefully if weights/network absent.

Run tests:

```bash
uv run pytest tests/ -v --tb=short
```

***

## 10. Code Style

- Python ≥ 3.13; use modern syntax (`from __future__ import annotations`).
- Line length: 100 characters.
- Use `ruff` for linting and formatting.
- Type hints are required on all public functions.
- Docstrings: Google style, with parameter and return type documentation.
- Imports: standard library → third-party → local (one group per blank line).
- No commented-out code in committed files.
- No `print()` in library code (`src/ald_sc/`); use logging or return values.
  `print()` is allowed in `scripts/` and `tests/`.

***

## 11. Key File Responsibilities

| File | Responsibility | Do Not |
|---|---|---|
| `arrow_prior.py` | Store and expose the frozen prior | Add trainable parameters |
| `audio_codec.py` | EnCodec encoder + baseline/graph decoders | Add training logic |
| `build_prior.py` | Construct the prior from embeddings | Import torch.nn modules |
| `data.py` | Audio data loading (ESC-50, folders, synthetic) | Add model logic |
| `dit.py` | 1-D DiT denoiser | Add VAE decoding logic |
| `dual_space.py` | DualSpaceMatrix M_N (2.5-D target) | Add audio-specific logic |
| `graph_decoder.py` | 1-D graph-structured decoder | Add diffusion logic |
| `losses.py` | Loss computation (L1+STFT, chart, smooth) | Import trainer or model modules |
| `sampling.py` | Inference samplers | Add training logic |
| `schedule.py` | Noise schedules | Import model modules |
| `spectral_schedule.py` | Per-mode entropic schedule | Import model modules |
| `trainer.py` | Training loops | Define model architectures |
| `wire_graph.py` | ArrowSpace adapter | Add model logic |

***

## 12. When Stuck

1. Re-read `docs/00.md` (research programme) and `docs/02.md` (audio
   adaptation) — these are the source of truth.
2. Check existing tests for usage patterns.
3. Check the upstream `arrowspace-latent-diffusion` for the image-domain
   reference implementation.
4. The frozen prior is **never** trainable. If you find yourself adding
   `nn.Parameter` to `arrow_prior.py`, you are on the wrong track.
5. The EnCodec encoder is **never** trainable. Only the decoder and DiT
   have trainable parameters.
6. Keep it simple. If a change adds significant complexity without
   testing the core hypothesis, defer it.
