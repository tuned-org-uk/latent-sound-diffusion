# Arrowspace Latent Diffusion (ALD-SC)

**ArrowSpace Latent Diffusion with Spectral Chart Conditioning** — a
spectrally conditioned latent diffusion model where a frozen ArrowSpace
graph-wiring prior defines a low-dimensional semantic manifold, a standard
spatial VAE latent carries local image detail, and the diffusion process
operates only on the spatial latent while the spectral chart provides
topology-aware global conditioning.

The architecture is deliberately simple: it reuses ESDM's frozen-prior
principle without replicating its full wave-recurrence and entropy-clock
machinery. It is developed alongside
[`entropic-semantic-diffusion`](https://github.com/tuned-org-uk/entropic-semantic-diffusion)
and follows the pedagogy of
[`arrowspace-diffusion-from-scratch`](https://github.com/tuned-org-uk/arrowspace-diffusion-from-scratch).

> **Hypothesis:** a frozen ArrowSpace spectral chart improves latent-image
> decoding and control at fixed latent capacity. The central falsifiable claim
> is that conditioning on feature-graph energy dispersion yields better
> global semantic coherence under compression — not merely better FID.

---

## The 2.5-D latent

Each image encodes to two coupled objects:

- **z** — a standard spatial VAE latent that carries local image detail.
- **A** — an F-channel semantic feature field over the latent plane, from
  which a compact spectral chart `c_spec` is derived via the frozen prior.

The ArrowSpace prior is built **once** from a corpus of feature embeddings
and never updated during training:

| Symbol | Shape | Description |
|---|---|---|
| `L_F` | (F, F) | Feature-space graph Laplacian |
| `U_q` | (F, q) | Leading q eigenvectors (smooth modes) |
| `eigvals_q` | (q,) | Corresponding eigenvalues |
| `lambdas_ed` | (F,) | Energy-dispersion distribution per feature node |
| `lambdas_chart` | (q,) | λ_ED projected onto the chart |

The spectral conditioner turns the per-image feature field into a compact
conditioning vector `c_spec` of shape `(3*q,)` = `[ẽ, λ_chart, ν]`, which
enters the DiT denoiser via AdaLN and the decoder via spectral gates.

```text
                          Frozen ArrowSpace prior
               ┌────────────────────────────────────┐
               │ L_F, U_q, Λ_q, λ_ED                │
               └────────────────────────────────────┘
                               │
 image x ── Encoder ──► spatial latent z ──► Latent DiT ──► ẑ
               │                    │                     │
               ▼                    │                     ▼
         feature field A             └──── spectral tokens ─┐
               │                                             │
               ▼                                             ▼
    project: A U_q U_q^T                          topology-conditioned
               │                                  VAE decoder
               ▼                                             │
    chart coordinates + band energy                           ▼
          s, e ──► c_spec ───────────────────────────────► image x̂
```

---

## Status

| Phase | Scope | State |
|---|---|---|
| Phase 1 | Spectral VAE (`arrow_prior`, `build_prior`, `vae`, `losses`, `trainer`) | in progress |
| Phase 2 | Latent diffusion (`dit`, `schedule`, `sampling`) | scaffolded — `dit.py` shipped (v0.0.1) |
| Phase 3 | Joint fine-tuning with spectral losses | planned |
| Phase 4 | Advanced ESDM concepts (entropy clock, wave recurrence) | future |

See [`AGENTS.md`](AGENTS.md) §7 for the full phase checklist and
[issue #1](https://github.com/tuned-org-uk/arrowspace-latent-diffusion/issues/1)
for the notebook-driven roadmap to image generation.

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management
and requires Python ≥ 3.13 with PyTorch ≥ 2.2.

```bash
git clone https://github.com/tuned-org-uk/arrowspace-latent-diffusion.git
cd arrowspace-latent-diffusion
uv sync
```

## Usage

```bash
# Run the test suite (CPU; no GPU required for boilerplate tests)
uv run pytest tests/ -v

# Lint and format
uv run ruff check src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/

# Scripts (landed incrementally per the roadmap)
uv run python scripts/train_vae.py      # Phase 1 entry point
uv run python scripts/train_diffusion.py  # Phase 2 entry point
uv run python scripts/sample.py         # image generation
```

Notebooks live under `notebooks/` and import library code from `src/ald_sc/`
(configured via `tool.pytest.ini_options.pythonpath` and an editable install).

---

## Repository layout

```
arrowspace-latent-diffusion/
├── pyproject.toml              # uv / hatchling project config
├── AGENTS.md                   # contributor guide (read this first)
├── configs/
│   ├── vae_base.yaml           # Phase 1 config
│   └── diffusion_base.yaml     # Phase 2 config
├── docs/
│   ├── 00.md                   # design document — Arrow-LDM
│   └── 01.md                   # design document — ALD-SC (ESDM transfer)
├── notebooks/                  # numbered milestones (see roadmap)
├── scripts/
│   ├── build_arrow_prior.py    # build frozen prior from embeddings
│   ├── train_vae.py            # Phase 1 training entry point
│   ├── train_diffusion.py      # Phase 2 training entry point
│   └── sample.py               # inference / image generation
├── src/ald_sc/
│   ├── __init__.py
│   ├── arrow_prior.py          # ArrowSpacePrior: frozen spectral prior
│   ├── build_prior.py          # build_arrow_prior() from corpus embeddings
│   ├── data.py                 # ImageFolderDataset, build_dataloader()
│   ├── dit.py                  # MinimalDiT: patchify + AdaLN transformer
│   ├── losses.py               # ALDSCLoss: diff + rec + chart + smooth
│   ├── sampling.py             # sample_euler(), sample_ddim()
│   ├── schedule.py             # CosineSchedule, LinearSchedule
│   ├── trainer.py              # train_vae(), train_diffusion()
│   └── vae.py                  # SpectralVAE: dual-head encoder + topology decoder
└── tests/
```

---

## Roadmap

The project is built one notebook at a time, each a vertical slice with TDD'd
modules + tests. Tracked in
[issue #1](https://github.com/tuned-org-uk/arrowspace-latent-diffusion/issues/1):

1. Noise schedule & forward corruption — [#2](https://github.com/tuned-org-uk/arrowspace-latent-diffusion/issues/2)
2. Frozen ArrowSpace prior & spectral chart — [#3](https://github.com/tuned-org-uk/arrowspace-latent-diffusion/issues/3)
3. Spectral VAE (encode/decode with topology) — [#4](https://github.com/tuned-org-uk/arrowspace-latent-diffusion/issues/4)
4. DiT denoiser & `c_spec` conditioning — [#5](https://github.com/tuned-org-uk/arrowspace-latent-diffusion/issues/5)
5. Latent diffusion training (Phase 2) — [#6](https://github.com/tuned-org-uk/arrowspace-latent-diffusion/issues/6)
6. Sampling & image generation (ship) — [#7](https://github.com/tuned-org-uk/arrowspace-latent-diffusion/issues/7)
7. *(stretch)* Joint fine-tuning & controllable editing (Phase 3) — [#8](https://github.com/tuned-org-uk/arrowspace-latent-diffusion/issues/8)

---

## Design constraints

- **Frozen prior.** `L_F`, `U_q` are buffers, never parameters. The graph
  defines the valid semantic geometry; learning happens on top of it.
- **Diffusion runs on `z` only.** No second diffusion process over the
  spectral chart `s` in v1.
- **Corpus-level prior, not per-image.** Do not construct a new graph per
  image.
- **Simple and clean.** Reuse ESDM concepts (frozen prior, projection,
  heat-kernel weights) rather than reimplementing the full vibrational system.

See [`AGENTS.md`](AGENTS.md) §6 for the full "what ALD-SC is / is not" and
§11 for per-file responsibilities.

---

## References

- Design documents: [`docs/00.md`](docs/00.md), [`docs/01.md`](docs/01.md)
- [Diffusion as spectral-geometric projection](https://www.tuned.org.uk/posts/021_diffusion_as_spectral_geometric_projection/) — theoretical background
- [`arrowspace-diffusion-from-scratch`](https://github.com/tuned-org-uk/arrowspace-diffusion-from-scratch) — pedagogical template
- [`entropic-semantic-diffusion`](https://github.com/tuned-org-uk/entropic-semantic-diffusion) — source of the frozen-prior concept
- [Chenyang Yuan — Diffusion models from scratch](https://chenyang.co/diffusion.html) (ICML 2024)
- [ArrowSpace — Spectral Search for Embeddings](https://doi.org/10.21105/joss.09002)
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (CVPR 2022)

## License

MIT
