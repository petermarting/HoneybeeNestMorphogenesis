# Model checkpoint manifest

The eight files in this directory are exactly the fitted models used in the
paper (Marting et al., *Current Biology* 2026,
[DOI 10.1016/j.cub.2026.10.064](https://doi.org/10.1016/j.cub.2026.10.064)).

## The paper's model set

The paper evaluates the masked-convolutional autoregressive model under every
choice of which spatial axes (depth `d`, height `h`, width `w`) enter the
receptive field, plus a spatial null model:

* **7 models** with kernel `k=3` — one for each of the 7 non-empty subsets of
  {depth, height, width}: three 1-D, three 2-D, and the full 3-D model.
  Larger kernels (`k` = 5…13) were tested in preliminary experiments but
  "dramatically increased processing time with little impact on prediction
  accuracy, so size k=3 was used" (Methods, "Masked-convolutional
  autoregressive model").
* **1 model** with `k=1` (no axes) — the spatial null model, shown as the
  dashed "no context" baseline in Figure 6.

Figure 6 reports the likelihoods with the configurations grouped by
dimensionality. Figure S11 reports the individual configurations.

All files load strictly (`weights_only=True`, strict `load_state_dict`) through
`combinator.models.load_any(path)` — the filename encodes the architecture, and
every file matches the geometry the `positional_embedding` weights fix:
input `(weeks, 20, 52, 104)`.

## Files

| File | Config key* | Axes in receptive field | Kernel k×k×k | Params | `positional_embedding` shape | Role in the paper |
|---|---|---|---|---:|---|---|
| `spatial_classifier_v2_k1_dFalse_hFalse_wFalse_posFalse_ctxFalse_int0.pth` | `(1, F, F, F, F, F, 0)` | — | 1×1×1 | 896 | — | Spatial null model (dashed baseline, Fig. 6) |
| `spatial_classifier_v2_k3_dTrue_hFalse_wFalse_posTrue_ctxTrue_int1.pth` | `(3, T, F, F, T, T, 1)` | depth | 3×1×1 | 85,376 | (1, 128, 20, 1, 1) | 1-D, depth |
| `spatial_classifier_v2_k3_dFalse_hTrue_wFalse_posTrue_ctxTrue_int1.pth` | `(3, F, T, F, T, T, 1)` | height | 1×3×1 | 89,472 | (1, 128, 1, 52, 1) | 1-D, height |
| `spatial_classifier_v2_k3_dFalse_hFalse_wTrue_posTrue_ctxTrue_int1.pth` | `(3, F, F, T, T, T, 1)` | width | 1×1×3 | 96,128 | (1, 128, 1, 1, 104) | 1-D, width |
| `spatial_classifier_v2_k3_dTrue_hTrue_wFalse_posTrue_ctxTrue_int1.pth` | `(3, T, T, F, T, T, 1)` | depth×height | 3×3×1 | 314,240 | (1, 128, 20, 52, 1) | 2-D, d×h |
| `spatial_classifier_v2_k3_dTrue_hFalse_wTrue_posTrue_ctxTrue_int1.pth` | `(3, T, F, T, T, T, 1)` | depth×width | 3×1×3 | 447,360 | (1, 128, 20, 1, 104) | 2-D, d×w |
| `spatial_classifier_v2_k3_dFalse_hTrue_wTrue_posTrue_ctxTrue_int1.pth` | `(3, F, T, T, T, T, 1)` | height×width | 1×3×3 | 873,344 | (1, 128, 1, 52, 104) | 2-D, h×w |
| `spatial_classifier_v2_k3_dTrue_hTrue_wTrue_posTrue_ctxTrue_int1.pth` | `(3, T, T, T, T, T, 1)` | depth×height×width | 3×3×3 | 14,320,512 | (1, 128, 20, 52, 104) | Full 3D model used in Figure 6 |

\* Key as stored by the evaluation code: `(k, d, h, w, pos, ctx, int)`. The
files in `results/` use the same names (`cfg_<that name>_hive_NN.npz`), and
`results/manifest.json` maps each result file back to its config key and
SHA-256.

> Note on the axis→dimension convention in the filename: the flags follow the
> order (d, h, w) = (depth, height, width) of the input tensor
> `(depth=20, height=52, width=104)`; e.g. the "width-only" model has
> `dFalse_hFalse_wTrue` and a 1×1×3 kernel.

## Training and evaluation workflow

* **Architecture & hyperparameters** (paper Methods): per-voxel class
  embedding (classes × 128) + learned per-class bias; one linear
  center-masked convolution (center voxel zeroed) with the kernel sizes above
  and zero padding; a learnable positional bias that varies only along the
  included axes; a first-order interaction term implemented with two 1×1×1
  pointwise convolutions so that `output = context + position + context2 +
  (context * position)`; a tied softmax head (`output @ embedding.T`) over the
  6 content classes. Optimizer AdamW (lr 1e-3, weight decay 1e-5,
  β=(0.9, 0.95)); the class-embedding and positional-bias parameter groups
  run at 10× the base learning rate; 5-epoch linear warmup then cosine
  annealing; 200 epochs; batch size 8; cross-entropy weighted by inverse
  class frequency. Training framework: PyTorch Lightning 2.5.0 (the version
  pinned in `pyproject.toml`); Python 3.12.3 per the paper.
* **Training data**: only the unshuffled control colonies, "123 unique 3D
  grids over all colony-weeks" (paper). The paper further states that grids
  were augmented by random width-axis flips (p = 0.5) because of the lack of
  stratification along the width axis.
* **Training entry points** — `combinator.train.train_spatial_classifier`
  trains one configuration and seeds the run with `pl.seed_everything(seed,
  workers=True)`. `combinator.train.train_all_v2` trains the eight configurations
  in `combinator.registry.V2_MODELS` with width-flip augmentation enabled.
* **Evaluation** — `combinator.eval.build_eval_hives` assembles six control
  colonies, six physically shuffled colonies, and six simulated shuffles of
  the controls. Each simulated shuffle keeps the two sides of a frame
  together and uses a seed from 42 through 47 with the default settings.
  `combinator.make_figures` evaluates the fitted checkpoints.
* **Geometry** — `combinator.preprocessing.process_grids` produces a
  `(depth, height, width)` grid of `(20, 52, 104)` per colony-week. The full
  3D checkpoint has a positional embedding of shape `(1, 128, 20, 52, 104)`.

## Using the checkpoints

Run the examples from the repository root after `uv sync --locked`.
This example evaluates the 3D checkpoint on one colony-week.

```python
from combinator import (
    build_eval_hives,
    evaluate_spatial_classifier,
    load_any,
    load_grids,
    map_all_labels,
    process_grids,
)

grids = process_grids(map_all_labels(load_grids("data/AllColoniesCombinedDS48v2.zip")))
hives = build_eval_hives(grids, num_control=6, seed=42)

device = "cpu"
checkpoint = "models/spatial_classifier_v2_k3_dTrue_hTrue_wTrue_posTrue_ctxTrue_int1.pth"
model = load_any(checkpoint, device=device)
results = evaluate_spatial_classifier(model, [hives[0][:1]], device=device)
```

Reproduce Figures 6 and S11 and their likelihood tables in `figures_out/`.

```bash
uv run python -m combinator.make_figures \
    --data data/AllColoniesCombinedDS48v2.zip --models-dir models \
    --outdir figures_out --device cpu
```

Use `--device cuda:0` with a CUDA-enabled PyTorch build. Verify the deposited
prediction files with the checksum manifest.

```bash
uv run python -m combinator.results_io verify results
```

## Optional training rerun

To retrain all eight configurations, pass the six preprocessed control
colonies to `train_all_v2`. Save the new checkpoints in a separate directory.

```python
from combinator import train_all_v2

train_all_v2(grids[:6], models_dir="retrained_models", seed=42, device="cuda:0")
```
