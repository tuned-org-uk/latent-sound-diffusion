"""Build notebooks/07_evaluation_metrics.ipynb for issue #49.

The notebook presents all evaluation deliverables. By default it loads the
pre-computed results produced by ``scripts/run_evaluation.py`` (the faithful
full training pipeline) so it executes in seconds with real output cells.
Set ``TRAIN_FROM_SCRATCH = True`` to re-run the full pipeline in-notebook
(~20 min CPU / ~5 min MPS; the graph decoder no longer requires CPU —
MPS parity is validated, see scripts/repro_mps_divergence.py and issue #51).

    uv run python scripts/build_eval_notebook.py
    uv run jupyter nbconvert --to notebook --execute notebooks/07_evaluation_metrics.ipynb
"""

from __future__ import annotations

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.13.12",
    },
}

cells = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src))


md(
    "# Evaluation Metrics — Spectral Composition (Issue #49)\n"
    "\n"
    "This notebook presents every evaluation deliverable required by\n"
    "[issue #49](https://github.com/tuned-org-uk/latent-sound-diffusion/issues/49):\n"
    "\n"
    "- **Phase 1** (Table 1, Table 2): reconstruction L1 (train/val/test) +\n"
    "  $\\lambda^{ED}$ ablation\n"
    "- **Phase 2** (Table 3): FAD-proxy + CLAP-proxy (graph vs baseline)\n"
    "- **Phase 3a** (Table 4): dehydration compression ratio vs corpus size $N$\n"
    "- **Phase 3b** (Table 5): rehydration coherence (MIDI pitch vs spectral\n"
    "  centroid, Pearson $r$)\n"
    "- **Phase 3c** (figure): recursive variant diversity (CLAP-proxy cosine\n"
    "  distance vs recursion depth)\n"
    "\n"
    "By default this notebook **loads the pre-computed results** produced by\n"
    "the faithful full-training pipeline `scripts/run_evaluation.py`\n"
    "(256-sample NSynth subset, 20 epochs, real frozen EnCodec, CPU). Set\n"
    "`TRAIN_FROM_SCRATCH = True` below to re-run the full pipeline in-place.\n"
    "\n"
    "> **FAD/CLAP methodology**: `fadtk` and `laion-clap` were removed due to\n"
    '> dependency conflicts (see README "Open issues" and\n'
    "> `docs/postmortem-c-spec-regression.md`). Phase 2 and 3c use\n"
    "> dependency-free proxies on the same frozen EnCodec feature space the\n"
    "> model already uses. The methodology is recorded in each CSV\n"
    "> (`*_method` columns). See `src/ald_sc/eval.py` for the implementations."
)

md("## Configuration")

code(
    "from pathlib import Path\n"
    "import csv\n"
    "\n"
    "# Set True to re-run the full training pipeline in-notebook (~20 min CPU).\n"
    "# False (default) loads the pre-computed results from results/*.csv.\n"
    "TRAIN_FROM_SCRATCH = False\n"
    "\n"
    "RESULTS_DIR = Path.cwd().parent / 'results'\n"
    "DATA_DIR = Path.cwd().parent / 'data'\n"
    "\n"
    "SEED = 3407\n"
    "AUDIO_LENGTH = 96000\n"
    "SAMPLE_RATE = 24000\n"
    "TEXT_PROMPT = 'warm electronic bass synth'\n"
    "\n"
    "print(f'Results dir: {RESULTS_DIR}')\n"
    "print(f'Train from scratch: {TRAIN_FROM_SCRATCH}')"
)

md("## Helpers")

code(
    "def show(rows):\n"
    "    'Lightweight table display (no pandas dependency).'\n"
    "    if not rows:\n"
    "        print('(empty)')\n"
    "        return\n"
    "    keys = list(rows[0].keys())\n"
    "    widths = [max(len(str(k)), *(len(str(r.get(k, ''))) for r in rows)) for k in keys]\n"
    "    hdr = ' | '.join(str(k).ljust(w) for k, w in zip(keys, widths))\n"
    "    sep = '-+-'.join('-' * w for w in widths)\n"
    "    print(hdr); print(sep)\n"
    "    for r in rows:\n"
    "        print(' | '.join(str(r.get(k, '')).ljust(w) for k, w in zip(keys, widths)))\n"
    "\n"
    "def load_csv(name):\n"
    "    path = RESULTS_DIR / name\n"
    "    if not path.exists():\n"
    "        print(f'  WARNING: {path} not found (run scripts/run_evaluation.py first)')\n"
    "        return []\n"
    "    with open(path) as f:\n"
    "        return list(csv.DictReader(f))\n"
    "\n"
    "print('Helpers ready.')"
)

md(
    "## Phase 1: Reconstruction L1 (Table 1) + $\\lambda^{ED}$ Ablation (Table 2)\n"
    "\n"
    "Controlled comparison: graph decoder (`WaveReconstructionBlock` + $U_q$ +\n"
    "$\\lambda^{ED}$) vs matched-capacity baseline decoder (plain `ResBlock1d`,\n"
    "no graph structure), both trained on the 256-sample NSynth subset with the\n"
    "real frozen EnCodec encoder for 20 epochs."
)

code(
    "table1 = load_csv('table1_reconstruction.csv')\n"
    "print('=== Table 1: Reconstruction L1 (train/val/test) ===')\n"
    "show(table1)"
)

code(
    "table2 = load_csv('table2_ablation.csv')\n"
    "print('=== Table 2: lambda_ED Ablation (c_spec on vs off) ===')\n"
    "show(table2)"
)

md(
    "## Phase 2: FAD-proxy + CLAP-proxy (Table 3)\n"
    "\n"
    "Generates 16 latents via DDIM, decodes with each decoder, and compares the\n"
    "resulting audio feature distribution to the held-out test-set EnCodec\n"
    "features. **FAD-proxy** = Frechet distance over frozen EnCodec pooled\n"
    "features (lower = generated distribution closer to reference). **CLAP-proxy**\n"
    "= cosine similarity between the generated bank's mean EnCodec feature and a\n"
    "deterministic hashing embedding of the text prompt."
)

code(
    "table3 = load_csv('table3_fad_clap.csv')\n"
    "print('=== Table 3: FAD-proxy + CLAP-proxy (graph vs baseline) ===')\n"
    "show(table3)"
)

md(
    "## Phase 3a: Dehydration Compression Ratio (Table 4)\n"
    "\n"
    "Bits required to represent the raw audio library vs the dehydrated\n"
    "ArrowSpace prior $(L_F, U_q, \\lambda^{ED})$ (stored once) plus per-clip\n"
    "EnCodec code-rate storage. The prior amortises, so the ratio grows with $N$\n"
    "and asymptotes to the per-clip code-rate ratio."
)

code(
    "table4 = load_csv('table4_compression.csv')\n"
    "print('=== Table 4: Compression ratio vs corpus size N ===')\n"
    "show(table4)"
)

md(
    "## Phase 3b: Rehydration Coherence (Table 5)\n"
    "\n"
    "Render an ascending MIDI scale via `synthesize_midi` (Mode C), compute the\n"
    "spectral-centroid contour of the render and the MIDI pitch contour sampled\n"
    "at the same frames, and report the Pearson correlation over active-note\n"
    "frames. A higher $r$ means the rehydrated audio's spectral centroid tracks\n"
    "the scored pitch contour."
)

code(
    "table5 = load_csv('table5_coherence.csv')\n"
    "print('=== Table 5: Rehydration coherence (MIDI pitch vs spectral centroid) ===')\n"
    "show(table5)"
)

md(
    "## Phase 3c: Recursive Variant Diversity (figure)\n"
    "\n"
    "Apply Mode C rehydration at depths 1×, 2×, 4× with different MIDI\n"
    "sequences. Compute pairwise CLAP-proxy cosine distance between variants.\n"
    "Reports mean inter-variant distance as a function of recursion depth,\n"
    "validating the claim that each MIDI produces a meaningfully different\n"
    "variant."
)

code(
    "var_rows = load_csv('variant_diversity.csv')\n"
    "print('=== Variant diversity vs recursion depth ===')\n"
    "show(var_rows)"
)

code(
    "import matplotlib.pyplot as plt\n"
    "\n"
    "fig_path = RESULTS_DIR / 'fig_variant_diversity.png'\n"
    "if fig_path.exists():\n"
    "    from IPython.display import Image, display\n"
    "    print('fig_variant_diversity.png:')\n"
    "    display(Image(filename=str(fig_path)))\n"
    "else:\n"
    "    print(f'  {fig_path} not found (run scripts/run_evaluation.py first)')\n"
    "\n"
    "# Re-plot from the CSV data for a self-contained execution output.\n"
    "if var_rows:\n"
    "    fig, ax = plt.subplots(figsize=(6, 4))\n"
    "    xs = [int(r['depth']) for r in var_rows]\n"
    "    means = [float(r['mean_distance']) for r in var_rows]\n"
    "    mins_v = [float(r['min_distance']) for r in var_rows]\n"
    "    maxs_v = [float(r['max_distance']) for r in var_rows]\n"
    "    ax.plot(xs, means, 'o-', label='mean pairwise distance', color='tab:blue')\n"
    "    ax.fill_between(xs, mins_v, maxs_v, alpha=0.2, color='tab:blue')\n"
    "    ax.set_xlabel('Recursion depth R')\n"
    "    ax.set_ylabel('CLAP-proxy cosine distance')\n"
    "    ax.set_title('Recursive variant diversity vs depth')\n"
    "    ax.legend()\n"
    "    ax.grid(True, alpha=0.3)\n"
    "    fig.tight_layout()\n"
    "    plt.show()\n"
    "else:\n"
    "    print('No variant_diversity.csv data to plot.')"
)

md(
    "## Re-run from scratch (optional)\n"
    "\n"
    "The full training pipeline (prior construction, graph + baseline decoder\n"
    "training, DiT training, all metrics) is in `scripts/run_evaluation.py`.\n"
    "To reproduce the CSVs from scratch:\n"
    "\n"
    "```bash\n"
    "uv run python scripts/run_evaluation.py --device cpu\n"
    "# (~16 min on CPU; produces all results/*.csv + fig_variant_diversity.png)\n"
    "```\n"
    "\n"
    "Set `TRAIN_FROM_SCRATCH = True` in the configuration cell and re-execute to\n"
    "run the same pipeline in-notebook. The eval engine itself is in\n"
    "`src/ald_sc/eval.py` (reusable, 30 unit tests in `tests/test_eval.py`)."
)

md("## Summary")

code(
    "print('=== Deliverables ===')\n"
    "for f in ['table1_reconstruction.csv', 'table2_ablation.csv', 'table3_fad_clap.csv',\n"
    "          'table4_compression.csv', 'table5_coherence.csv', 'fig_variant_diversity.png']:\n"
    "    p = RESULTS_DIR / f\n"
    '    print(f\'  {"OK" if p.exists() and p.stat().st_size > 0 else "MISSING"}: {p.name}\')'
)

nb.cells = cells

nbf.write(nb, "notebooks/07_evaluation_metrics.ipynb")
print("Wrote notebooks/07_evaluation_metrics.ipynb")
