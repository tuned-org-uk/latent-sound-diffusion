# arrowspace-latent-diffusion

Name: Arrowspace Latent Diffusion

## 1. Project Identity

**ALD-SC** (ArrowSpace Latent Diffusion with Spectral Chart Conditioning) is a
spectrally conditioned latent diffusion model. A frozen ArrowSpace graph-wiring
prior (L_F, U_q, Λ_q, λ_ED) defines a low-dimensional semantic manifold. A
standard spatial VAE latent carries local image detail. The diffusion process
operates only on the spatial latent; the spectral chart provides global
topology-aware conditioning.

The architecture is deliberately simple: it reuses ESDM's frozen-prior
principle without replicating its full wave-recurrence and entropy-clock
machinery. It reuses concepts developed in https://github.com/tuned-org-uk/entropic-semantic-diffusion

### 1.1 The research programme: decoding on the feature-space manifold

The basic point of this research programme is to design **decoding using
three structures**, all computed from the training corpus via the
ArrowSpace library (https://github.com/tuned-org-uk/pyarrowspace):

1. **The item-space** — the spatial latent z carrying local image detail.
2. **The feature-space graph Laplacian** L_F — its eigenvectors U_q define
   the smooth semantic subspace; its eigenvalues ν_k define entropy
   exchange rates.
3. **The dispersion network** λ_ED — ArrowSpace's per-feature
   energy-dispersion distribution (https://arxiv.org/abs/2606.21535).
   This is not a diagnostic; it is a representation of how semantic
   structure is distributed over the feature graph.

The 2.5-D space is the structure defined by the projection of the
item-space into the feature-space graph Laplacian (originally the
DualSpaceMatrix M_N = α‖VVᵀᵗ‖_F − β‖V L_F Vᵀᵗ‖_F in ESDM). This is the
training dataset for encoding.

**Decoding must use L_F and λ_ED as constructive elements of the
decoding operator, not merely as conditioning signals.** L_F defines the
reconstruction paths (which directions to reconstruct along);
λ_ED defines the energy allocation (where to concentrate reconstruction
effort). Standard VAEs decode via the reparameterization trick (a
continuous surface integral over a Gaussian latent). The research target
is to find an analogous construction using the graph structures.

**The Barontini entropic clock and the ESDM vibrational harness are
tools** for achieving this decoding design, not separate concerns:
- The clock provides temporal dynamics for decoding (when modes resolve,
  intrinsic stopping via heat death).
- The vibrational harness provides wave-based reconstruction on the graph
  (propagating information along smooth directions, weighted by the
  dispersion network).

See `docs/00.md` § "The research programme" for the full design
statement. The central falsifiable claim is that decoding on the
feature-space manifold yields better global semantic coherence under
compression than decoding on an unconstrained ambient latent.

***

## 2. Repository Layout

```
arrowspace-latent-diffusion/
├── pyproject.toml              # uv / hatchling project config
├── README.md
├── AGENTS.md                   # this file
├── .gitignore
├── configs/
│   ├── vae_base.yaml           # Phase 1 config
│   └── diffusion_base.yaml     # Phase 2 config
├── docs/
│   └── 00.md                   # design document
│   └── 01.md                   # design document
├── scripts/
│   ├── build_arrow_prior.py    # build frozen prior from embeddings
│   ├── train_vae.py            # Phase 1 training entry point
│   ├── train_diffusion.py     # Phase 2 training entry point
│   └── sample.py               # inference / image generation
├── src/ald_sc/
│   ├── __init__.py
│   ├── arrow_prior.py          # ArrowSpacePrior: frozen spectral prior
│   ├── build_prior.py           # build_arrow_prior() from corpus embeddings
│   ├── data.py                  # ImageFolderDataset, build_dataloader()
│   ├── dit.py                   # MinimalDiT: patchify + AdaLN transformer
│   ├── losses.py                # ALDSCLoss: diff + rec + chart + smooth
│   ├── sampling.py              # sample_euler(), sample_ddim()
│   ├── schedule.py              # CosineSchedule, LinearSchedule
│   ├── trainer.py               # train_vae(), train_diffusion()
│   └── vae.py                   # SpectralVAE: dual-head encoder + topology decoder
└── tests/
    ├── __init__.py
    ├── test_arrow_prior.py
    ├── test_dit.py
    ├── test_losses.py
    ├── test_schedule.py
    └── test_vae.py
```

***

## 3. Environment Setup

This project uses **uv** for dependency management.

```bash
# Clone and install
git clone https://github.com/tuned-org-uk/arrowspace-latent-diffusion.git
cd arrowspace-latent-diffusion
uv sync

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/ scripts/
```

Always use the local arrowspace-latent-diffusion/.venv environment.

Python ≥ 3.13 is required. PyTorch ≥ 2.2 is the primary dependency.

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

### 4.2 The 2.5-D Latent

Each image encodes to two coupled objects:

- **z ∈ R^{B×c×h×w}** — spatial latent (standard VAE latent)
- **A ∈ R^{B×N×F}** — feature field over the latent plane

The ArrowSpace projection restricts A to the smooth subspace. The spectral
chart provides compact global conditioning via `c_spec`.

### 4.3 Noise Schedule

- `CosineSchedule` — ScheduleLDM-style, for diffusion in VAE-latent space
- `LinearSchedule` — ScheduleDDPM-style, for comparison

Both support `add_noise(z0, t, noise)` and `v_target(z0, t, noise)` for
v-prediction training.

### 4.4 Loss Components

| Loss | Formula | When active |
|---|---|---|
| L_diff | \|\|v − v_θ(z_t, t, c_spec)\|\|² | Phase 2 |
| L_rec | \|\|x − x̂\|\|₁ + λ_perc·LPIPS | Phase 1, Phase 3 |
| L_chart | \|\|ẽ(x) − ẽ(x̂)\|\|² | Phase 1, Phase 3 |
| L_smooth | \|\|A(I − U_qU_q^T)\|\|²_F / \|\|A\|\|²_F | Phase 1, Phase 3 |

***

## 5. Development Workflow

### 5.1 Standard Development Cycle

1. **Understand the design**: Read `docs/00-DESIGN.md` first.
2. **Run tests**: `uv run pytest tests/ -v` — all must pass before pushing.
3. **Lint**: `uv run ruff check src/ tests/ scripts/` — fix all warnings.
4. **Format**: `uv run ruff format src/ tests/ scripts/`
5. **Commit**: Use conventional commit messages (see §8).

### 5.2 Adding a New Module

1. Create the file in `src/ald_sc/`.
2. Export public symbols in `src/ald_sc/__init__.py`.
3. Add corresponding test file in `tests/`.
4. Update `docs/00-DESIGN.md` if the architecture changes.
5. Run `uv run pytest tests/ -v` and `uv run ruff check`.

### 5.3 Modifying an Existing Module

1. Read the module docstring and existing tests first.
2. Make the minimal change needed.
3. Update or add tests to cover the change.
4. Run tests and lint.
5. If the change affects the architecture, update `docs/00-DESIGN.md`.

***

## 6. Architecture Decisions and Constraints

### 6.1 What ALD-SC Is

- A latent diffusion model (like Stable Diffusion) with an additional frozen
  spectral conditioning path derived from ArrowSpace graph wiring.
- The diffusion backbone is a minimal DiT; swap for a larger architecture when
  scaling up.
- The VAE has a dual-head encoder (spatial + feature) and a topology-adaptive
  decoder with spectral-chart gating.

### 6.2 What ALD-SC Is Not

- **Not a full ESDM.** The wave recurrence, entropy clock, Rayleigh-gradient
  restoring force, and learned pump are intentionally deferred. They can be
  added as isolated second-stage contributions once the core hypothesis is
  validated.
- **Not a per-image graph.** The ArrowSpace prior is corpus-level and frozen.
  Do not construct a new graph per image.
- **Not a coupled dual-diffusion.** Diffusion runs on z only. Do not run a
  second diffusion process on the spectral chart in v1.

### 6.3 Design Principles

1. **Frozen prior**: L_F, U_q are buffers, not Parameters. The graph defines
   the valid semantic geometry; learning happens on top of it.
2. **Simple and clean**: The architecture should remain readable. Avoid adding
   complexity that is not needed to test the central hypothesis.
3. **Reusable concepts**: Import and reuse ESDM concepts where they fit
   (frozen prior, projection, heat-kernel weights) rather than reimplementing
   the full vibrational system.
4. **Convention over configuration**: Use the provided YAML configs as
   defaults. Override via CLI args or config edits.

***

## 7. Implementation Phases

### Phase 1: Spectral VAE (current focus)

**Goal**: Demonstrate that a topology-conditioned VAE improves reconstruction
or latent interpolation at fixed capacity.

- [x] `arrow_prior.py` — ArrowSpacePrior container
- [x] `build_prior.py` — prior construction from embeddings
- [x] `vae.py` — SpectralVAE with dual-head encoder + topology decoder
- [x] `losses.py` — L_rec + L_chart + L_smooth
- [x] `trainer.py` — `train_vae()` loop
- [x] `scripts/train_vae.py` — Phase 1 entry point
- [x] `scripts/build_arrow_prior.py` — prior builder script
- [ ] Run on CIFAR-10 or equivalent with DINO/SigLIP embeddings
- [ ] Compare against baseline VAE (no spectral conditioning)
- [ ] Measure: PSNR, SSIM, LPIPS, chart-energy error, off-manifold ratio

### Phase 2: Latent Diffusion

**Goal**: Train the DiT denoiser conditioned on spectral chart tokens.

- [x] `dit.py` — MinimalDiT with AdaLN conditioning
- [x] `schedule.py` — cosine and linear schedules
- [x] `trainer.py` — `train_diffusion()` loop
- [x] `scripts/train_diffusion.py` — Phase 2 entry point
- [ ] Run with frozen Phase 1 VAE
- [ ] Measure: FID, spectral consistency of generated images
- [ ] Ablation: with vs. without spectral conditioning

### Phase 3: Joint Fine-tuning

**Goal**: End-to-end optimisation with low-weight spectral losses.

- [ ] Unfreeze decoder adapters
- [ ] Train with full loss (L_diff + L_rec + L_chart + L_smooth)
- [ ] Validate that spectral structure is preserved under generation

### Phase 4: Advanced ESDM Concepts (future)

- [ ] Add entropy-clock-gated schedule
- [ ] Add wave-recurrence encoder block
- [ ] Add Rayleigh-gradient regularizer
- [ ] Explore coupled z + s diffusion

***

## 8. Commit Message Conventions

Use conventional commits:

```
feat: add spectral gating to decoder block 3
fix: correct chart_energy_descriptor padding when q < F
refactor: move reparametrize to static method
test: add off_manifold_energy boundary tests
docs: update 00-DESIGN.md with Phase 2 details
chore: bump torch to 2.3
```

***

## 9. Testing Requirements

- Every public function and class must have at least one test.
- Tests must pass on CPU (no GPU required for boilerplate tests).
- Use small tensor sizes in tests (e.g., F=32, q=8, img_size=64).
- Test both forward pass shapes and gradient flow.

Run tests:

```bash
uv run pytest tests/ -v --tb=short
```

***

## 10. Code Style

- Python ≥ 3.11; use modern syntax (`from __future__ import annotations`).
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
| `build_prior.py` | Construct the prior from embeddings | Import torch.nn modules |
| `vae.py` | Encoder + topology decoder | Add diffusion logic |
| `dit.py` | DiT denoiser | Add VAE decoding logic |
| `schedule.py` | Noise schedules | Import model modules |
| `losses.py` | Loss computation | Import trainer or model modules (except for type hints) |
| `trainer.py` | Training loops | Define model architectures |
| `sampling.py` | Inference samplers | Add training logic |
| `data.py` | Data loading | Add model logic |

***

## 12. When Stuck

1. Re-read `docs/00.md` and `docs/01.md` — the design doc is the source of truth.
2. Check existing tests for usage patterns.
3. Check the ESDM design doc for the original concept definitions.
4. The frozen prior is **never** trainable. If you find yourself adding
   `nn.Parameter` to `arrow_prior.py`, you are on the wrong track.
5. The diffusion model operates on **z only**. If you are adding a second
   diffusion process on spectral coefficients, you are ahead of the v1 scope.
6. Keep it simple. If a change adds significant complexity without testing the
   core hypothesis (spectral conditioning improves latent diffusion), defer it
   to Phase 4.
'''
