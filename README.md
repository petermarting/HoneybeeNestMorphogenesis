# Honeybee Nest Morphogenesis — data and code

Analysis code, model weights, and data for the honeybee nest
4D atlas study in

> **Marting PR, Prathibha P, Koger B, Rogers SR, Graving JM, Smith ML.**
> *Morphogenesis, differentiation, and robust multi-scale organization of the
> honeybee nest revealed by a novel 4D atlas.*
> **Current Biology**. Published online November 18, 2026.
> [DOI 10.1016/j.cub.2026.10.064](https://doi.org/10.1016/j.cub.2026.10.064)

This repository contains

* **`combinator/`** — the Python package for fitting and evaluating the
  autoregressive model and regenerating Figures 6 and S11.
* **`models/`** — the eight fitted paper model checkpoints
  (k=3 across all 7 one/two/three-axis configurations, plus the spatial
  null-model baseline, ≈62 MB total) with per-file provenance in
  [`models/MANIFEST.md`](models/MANIFEST.md).
* **`data/`** — the model's detailed 21-class colony array
  (`AllColoniesCombinedDS48v2.zip`, ~3.6 MB; zipped because the raw 156 MB
  `.npy` exceeds GitHub's 100 MB/file cap).
* **`results/`** — 144 cached prediction files (about 398 MiB) with
  ground truth, predictions, entropy, and log-likelihood for eight models
  across 18 evaluation hives. The [manifest](results/manifest.json) records
  the SHA-256 checksum for each file.
* **`figures/`** — the Figure 6 likelihood plots and 3D prediction example,
  Figure S11, the receptive-field legend, and the likelihood CSVs.
* **`combinator/content/`** — the observational analyses
  for Figures 1–5, organized into modules for seasonality, organization
  at the cell, comb, and nest levels, specialization, and the shuffling
  experiment. [Run the observational analyses](#reproduce-the-observational-analyses)
* **`CombContent_Make.ipynb`** / **`CombContent_Analysis.ipynb`** — package-driven
  notebooks with saved figure outputs and table checks.
* **`tests/`** — checks for reported statistics, rebuilt table rows, and
  figure generation.
* **`figures/analysis/`** — 20 vector PDFs covering the supported panels
  across Figures 1–5. [Reproduction coverage](RESULTS.md)
* **`autoregressive_model.ipynb`** — the model notebook, driven entirely by
  the `combinator` package. It walks through preprocessing, evaluation of
  the fitted checkpoints, and rendering Figures 6 and S11. It also includes
  a two-epoch training smoke test and an optional `train_all_v2` recipe.
  The figure outputs use the fitted checkpoints or their cached predictions.

---

## Installation

The package requires Python 3.11 or later. Run the commands from the
repository root. [uv](https://github.com/astral-sh/uv) installs the package
in an editable environment using [the lock file](uv.lock).

```bash
uv sync --locked
```

The `analysis` extra supplies dependencies for the comb-content workflows.
Install it with `uv sync --locked --extra analysis`.

## Quick start

Run the Python example with `uv run python` to evaluate the fitted 3D
checkpoint on one colony-week. Set `device = "cuda:0"` to use a
CUDA-enabled PyTorch build.

```python
from combinator import (
    build_eval_hives,
    evaluate_spatial_classifier,
    load_any,
    load_grids,
    map_all_labels,
    process_grids,
)

grids = load_grids("data/AllColoniesCombinedDS48v2.zip")
grids = process_grids(map_all_labels(grids))
hives = build_eval_hives(grids, num_control=6, seed=42)

device = "cpu"
checkpoint = "models/spatial_classifier_v2_k3_dTrue_hTrue_wTrue_posTrue_ctxTrue_int1.pth"
model = load_any(checkpoint, device=device)
example = [hives[0][:1]]
results = evaluate_spatial_classifier(model, example, device=device)
print(results["hives"][0]["predictions"].shape)
```

### Reproduce Figures 6 and S11 from the fitted weights

The command writes the figures and likelihood tables to `figures_out/`.

```bash
uv run python -m combinator.make_figures \
    --data data/AllColoniesCombinedDS48v2.zip \
    --models-dir models --outdir figures_out --device cpu
```

Pass `--device cuda:0` for GPU inference. The default `paper` palette uses
seaborn's colorblind colors with wood in dark red. `--palette seaborn` uses
seaborn's original colors and `--palette content` uses the Figures 1–5 palette.

### Verify and use the cached predictions

```bash
uv run python -m combinator.results_io verify results
```

To regenerate the figures and tables from the cache, run the following
with `uv run python`.

```python
from combinator import (
    load_results,
    make_figures,
    write_figure6_table,
    write_per_config_table,
)

cached = load_results("results")
outdir = "figures_out/cached"
make_figures(cached, outdir)
write_figure6_table(cached, outdir)
write_per_config_table(cached, outdir)
```

The cache stores entropy and log-likelihood in `float16`. Recomputing
both committed CSV summaries from the cached likelihoods agreed within
`0.0001` at their displayed precision.

The model notebook offers the same cache workflow through `USE_CACHED_RESULTS`.
Open it from the repository root with Jupyter.

```bash
uv run --with notebook jupyter notebook autoregressive_model.ipynb
```

## Data and provenance

The model is trained and evaluated on the **detailed** per-colony grids.
These are produced by the comb-image segmentation pipeline and (in their
raw, un-aligned form) are large, so the model input is committed **zipped**
under [`data/`](data/) — `zip -9` reduces the 156 MB `.npy` to ~3.6 MB.
`load_grids()` accepts a `.zip` containing a single `.npy` directly, so no
download or unzip step is needed.

| File | Shape per colony (weeks×D×H×W) | Classes | Where | Purpose |
|---|---|---|---|---|
| `AllColoniesCombinedDS48v2.npy` (zipped) | (weeks, 20, 77, 116) | detailed (0–19, 255) | in this repo, as `data/AllColoniesCombinedDS48v2.zip` | **model** input; the 21-class DS48v2 masks (12 colonies, 228 weeks). Remapped to 6 classes, aligned, and cropped by the pipeline to (weeks, 20, 52, 104). The raw file is 156 MB (over GitHub's 100 MB/file cap), so it is committed zipped; `load_grids()` reads the zip directly |
| `CombMasks_Contents_aligned.npy` | (weeks, 20, 52, 100) | 0–10 | in this repo | deposited; comb-**content** masks (already aligned) |
| `CombMasks_CombTypes_aligned.npy` | (weeks, 20, 52, 100) | 0–5 | in this repo | deposited; comb-**type** labels (already aligned) |
| `CombMasks_Combined_aligned.npy` | (weeks, 20, 52, 100) | 0–19 | in this repo | deposited; content + type combined (already aligned) |
| `CombMasks_Contents_simple.npy` | (weeks, 20, 77, 116) | 0–4, 255 | in this repo | deposited; simplified content masks, unaligned |
| `CombMasks_Contents_crop.npy` | (weeks, 20, 42, 88) | 0–10 | in this repo | deposited; content masks, a tighter crop |
| `df_*.csv` | — | — | in this repo | per-cell / per-colony statistics derived from the masks, consumed by `CombContent_Analysis.ipynb` |
| `models/*.pth` | — | — | in this repo | eight fitted checkpoints (see [`models/MANIFEST.md`](models/MANIFEST.md)) |
| `results/*.npz` | — | — | in this repo | per-hive model outputs (see `results/manifest.json`) |

The model uses the detailed 21-class DS48v2 masks. The observational
workflows use the released `CombMasks_*.npy` arrays and their derived
`df_*.csv` tables, which have their own label taxonomies and alignment.

### Note on grid geometry

The positional-embedding weights of the committed checkpoints fix the model
input geometry to **(depth, height, width) = (20, 52, 104)**: 20 = 10
combined frame-sides (Side A + Side B of each of the 10 frames, stored
interleaved), and 52 × 104 is the mean-wood bounding-box crop (margin 1,
threshold 0.5) of the aligned 77 × 116 comb image — exactly what
`combinator.preprocessing.process_grids` computes. The weights (e.g. the
`positional_embedding` shape `(1, 128, 20, 52, 104)`) are authoritative and
the code matches them exactly; loading is strict (`weights_only=True`,
`load_state_dict` strict).

## Reproduce the observational analyses

The analysis pipeline reads the released `CombMasks_*.npy` arrays and
`df_*.csv` tables. Run the figure generator and the fast tests from the
repository root.

```bash
uv sync --locked --extra analysis --extra dev
uv run python -m combinator.make_analysis_figures --outdir figures_out/analysis
uv run pytest -q
```

Use `--figs 1,3` to render a subset of figures. Run `uv run pytest -q -m slow`
for the longer bootstrap, figure, and table checks. GitHub Actions runs
Ruff and the fast tests on pushes and pull requests.

### Rebuild derived tables

Write the rebuilt tables to a separate directory so the released CSVs
remain the reference inputs for the figures and tests.

```bash
uv run python -m combinator.make_content_tables \
    --tables nearestdist --outdir rebuilt-tables
```

The available generators are `cellneib`, `cellfuture`, and `nearestdist`.
The checks compare rebuilt rows with the released tables using tolerances
for floating-point and Monte Carlo differences.

### Reproduction coverage

The tests cover reported values including week-10 composition, cluster
metrics, cell specialization, and brood failure. Remaining gaps include
the segmentation step behind Figure 1B, Figure 4C, some KS p-values,
polarization and comb-specialization statistics, and missing inputs for
some derived tables.
The figure-to-code map lists the validation results and gaps for each
panel. [RESULTS.md](RESULTS.md)

## Model

`SpatialClassifier` is a per-voxel autoregressive content classifier.
See the implementation in [`combinator/models.py`](combinator/models.py).

* **Class embedding**: a learned (classes × 128) table maps each voxel's
  integer label to a 128-d vector; a per-class bias is added.
* **Center-masked conv**: a single linear convolution with kernel
  (k, k, k), with `k` = 3 on *included* axes and 1 on excluded axes, whose
  center voxel is hard-zeroed so the true label never appears in the input.
* **Positional bias**: a learnable (1, 128, D', H', W') field, full size on
  included axes and 1 on excluded axes, broadcast over the grid.
* **First-order interaction**: a learned "context × position" term — two
  1×1×1 pointwise convolutions produce two feature vectors that are
  multiplied element-wise and added back to `context + position`
  (i.e. `output = context + position + context2 + (context * position)`).
* **Tied head**: logits = `output @ class_embedding.T`, softmaxed to a
  distribution over the 6 classes.

The training recipe uses AdamW with a base learning rate of `1e-3`,
weight decay of `1e-5`, and betas `(0.9, 0.95)`. The class embedding and
positional parameters use 10 times the base learning rate. The 200-epoch
schedule uses five warmup epochs followed by cosine annealing, with a batch
size of eight and inverse-frequency class weights.

After preprocessing, pass the six controls (`grids[:6]`) to
`combinator.train_all_v2` to retrain all eight configurations. Set
`models_dir="retrained_models"` to write the new checkpoints there.
The recipe uses width-flip augmentation during training.
[Checkpoint and training details](models/MANIFEST.md)

### The 8 committed checkpoints

| File (see `models/` for full names) | Role in the paper |
|---|---|
| `..._v2_k1_..._int0` | Spatial null model (k=1, no axes) — the "no context" baseline (dashed line) |
| 3 × 1D: w, h, d | Receptive field on a single axis |
| 3 × 2D: d+h, d+w, h+w | Two axes |
| 1 × 3D: d+h+w | Full 3-D context — the main model |

Figure 6 groups the spatial configurations by dimensionality, bootstraps
pooled colony-week likelihoods within each group, and uses the fitted
`k=1` model as a baseline. [Figure 6](figures/control_physical_simulated_likelihood.pdf)
and [likelihood table](figures/mean_likelihood_figure6.csv)

Figure S11 shows the seven spatial configurations individually, with a
colony-then-week bootstrap and a chance baseline of `1/6`.
[Figure S11](figures/supplement_config_likelihood.pdf)
and [likelihood table](figures/mean_likelihood_per_config.csv)

## What "shuffling" means

* **Control** — DD1, DD2, DD4, DD5, DD6, and DD7, with frames in their
  natural order.
* **Physically shuffled** — SH2 through SH7, whose frames were moved
  within the hive during the experiment.
* **Simulated** — copies of the six controls with their ten frame pairs
  permuted along the depth axis. The sides of each frame stay together.
  `build_eval_hives(..., seed=42)` builds these permutations reproducibly.

## License

MIT (see `LICENSE`). Code and weights; data files retain their original
provenance and sharing terms (see the table above).
