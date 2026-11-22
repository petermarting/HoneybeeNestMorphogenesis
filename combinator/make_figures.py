"""Regenerate the model figures and likelihood tables.

Run from the repository root.

    uv run python -m combinator.make_figures \
        --data data/AllColoniesCombinedDS48v2.zip --models-dir models \
        --outdir figures_out --device cpu

The workflow loads the fitted checkpoints and evaluates six control hives,
six physically shuffled hives, and six simulated shuffles of the controls.
It writes Figures 6 and S11, the receptive-field legend, and two likelihood
tables. The ``--save-results PATH`` option writes a PyTorch ``.pt`` file.
``combinator.results_io`` reads the released per-hive ``.npz`` cache.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .dataset import load_grids
from .eval import build_eval_hives, evaluate_spatial_classifier
from .labels import map_all_labels
from .models import from_config
from .preprocessing import process_grids
from .registry import V2_MODELS, v2_config
from . import plotting


def evaluate_all_v2(hive_grids, models_dir, device):
    """Evaluate every committed v2 model; returns {key: eval_output}."""
    import torch
    results = {}
    for fname, key in V2_MODELS:
        path = os.path.join(models_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        model = from_config(v2_config(key), device=device)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"  evaluating {fname} ...")
        results[key] = evaluate_spatial_classifier(
            model, hive_grids, device=device, batch_size=16)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return results


def make_figures(results, outdir, hive_idx=4, week=10):
    """Reproduce the three committed figures from a full results dict.

    ``hive_idx``/``week`` default to 4/10 to match the notebook run that
    produced the originally committed figures."""
    os.makedirs(outdir, exist_ok=True)
    model_keys = list(results)

    pdf = plotting.plot_control_physical_simulated(
        results, model_keys, num_classes=6, class_names=plotting.DEFAULT_NAMES,
        n_bootstrap=1000, ci=95, random_state=42,
        out=os.path.join(outdir, "control_physical_simulated_likelihood.pdf"))
    print(f"  wrote {pdf}")

    # Figure 6B uses the full 3D model (the notebook used model_keys[0],
    # which in the original evaluation order was the (T,T,T) 3D config).
    d3d_key = next(k for k in model_keys
                   if k[0] == 3 and k[1:4] == (True, True, True))
    png = plotting.plot_model_scatter3d_simplified(
        results, model_keys, model_idx=model_keys.index(d3d_key),
        hive_idx=hive_idx, week=week,
        out=os.path.join(outdir, "3d_predictions_likelihood.png"))
    print(f"  wrote {png}")

    legend = plotting.plot_kernel_legend_subset(
        out=os.path.join(outdir, "spatial_legend_subset.pdf"))
    print(f"  wrote {legend}")

    config_pdf = plotting.plot_control_physical_simulated_full(
        results, model_keys, num_classes=6, class_names=plotting.DEFAULT_NAMES,
        n_bootstrap=1000, ci=95, random_state=42,
        out=os.path.join(outdir, "supplement_config_likelihood.pdf"))
    print(f"  wrote {config_pdf}")


def make_figures_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="path to the colony grids (.npy or .zip containing a single .npy)")
    ap.add_argument("--models-dir", default="models",
                    help="directory containing the committed .pth files")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--device", default="cuda:0",
                    help="torch device (select the GPU with "
                         "CUDA_VISIBLE_DEVICES, e.g. CUDA_VISIBLE_DEVICES=1)")
    ap.add_argument("--hive-idx", type=int, default=4)
    ap.add_argument("--week", type=int, default=10)
    ap.add_argument("--palette", default="paper",
                    choices=plotting.CLASS_PALETTE_MODES,
                    help="'paper' (default): the published model-figure colors "
                         "(seaborn 'colorblind', wood in red); 'seaborn': the "
                         "default seaborn 'colorblind' palette; 'content': the "
                         "nest-content palette of Figures 1-5")
    ap.add_argument("--save-results", default=None,
                    help="optional path (torch .pt) to save the full results dict")
    args = ap.parse_args()
    plotting.set_class_palette_mode(args.palette)

    print(f"Loading data {args.data} ...")
    grids = load_grids(args.data)
    grids = map_all_labels(grids)
    print(f"Aligning and cropping {len(grids)} colonies ...")
    grids = process_grids(grids)

    print("Building 18-hive evaluation set ...")
    all_hives = build_eval_hives(grids, num_control=6, seed=42)

    device = args.device
    print(f"Evaluating {len(V2_MODELS)} models on {device} ...")
    results = evaluate_all_v2(all_hives, args.models_dir, device)

    make_figures(results, args.outdir, hive_idx=args.hive_idx, week=args.week)

    if args.save_results:
        _save_results(results, args.save_results)

    # Compact per-class mean-likelihood table (the Figure-6 numbers).
    write_figure6_table(results, args.outdir)

    # Per-configuration table (the supplemental-figure numbers).
    write_per_config_table(results, args.outdir)


def write_figure6_table(results, outdir):
    """Write mean_likelihood_figure6.csv: per-class mean likelihood (mean
    over the configurations in each dimensionality group), by path, plus the
    k=1 baseline.  This is the compact table behind Figure 6."""
    import csv
    import os

    class_names = plotting.DEFAULT_NAMES
    path_slices = {"control": slice(0, 6), "treatment": slice(6, 12),
                   "simulated": slice(12, 18)}
    groups = {
        "1D": [(True, False, False), (False, True, False), (False, False, True)],
        "2D": [(True, True, False), (True, False, True), (False, True, True)],
        "3D": [(True, True, True)],
    }
    path_names = list(path_slices)

    def mean_likelihood(hives, cls):
        """Mean, over weeks, of the mean predicted probability of class
        ``cls`` at voxels whose ground truth label is ``cls``."""
        vals = []
        for hive in hives:
            gt = hive["ground_truth"]
            probs = np.exp(hive["log_likelihood"])
            for t in range(gt.shape[0]):
                m = gt[t].flatten() == cls
                if np.any(m):
                    vals.append(probs[t].reshape(-1)[m].mean())
        return float(np.mean(vals)) if vals else float("nan")

    baseline_key = next(k for k in results
                        if k[0] == 1 and k[1:4] == (False, False, False))
    baseline = {
        name: {cls: mean_likelihood(results[baseline_key]["hives"][sl], cls)
               for cls in range(len(class_names))}
        for name, sl in path_slices.items()
    }

    with open(os.path.join(outdir, "mean_likelihood_figure6.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        header = ["dimensionality", "class"] + path_names + \
                 [f"baseline_{p}" for p in path_names]
        w.writerow(header)
        for dim, dhws in groups.items():
            for cls, cname in enumerate(class_names):
                vals = {}
                for name, sl in path_slices.items():
                    per_config = [mean_likelihood(results[key]["hives"][sl], cls)
                                   for key in results
                                   if key[0] == 3 and key[1:4] in dhws]
                    per_config = [v for v in per_config if not np.isnan(v)]
                    vals[name] = float(np.mean(per_config)) if per_config else float("nan")
                row = [dim, cname] + \
                      [f"{vals[p]:.4f}" for p in path_names] + \
                      [f"{baseline[p][cls]:.4f}" for p in path_names]
                w.writerow(row)
    print(f"  wrote {os.path.join(outdir, 'mean_likelihood_figure6.csv')}")


def write_per_config_table(results, outdir):
    """Write mean_likelihood_per_config.csv: per-configuration bootstrap mean
    likelihood for each individual k=3 configuration (rows = class x
    configuration), by hive group, plus the pure-chance baseline -- the means
    the per-configuration supplemental figure plots."""
    import csv

    from .plotting import (CONFIG_LABELS, CONFIG_ORDER,
                           per_config_bootstrap_stats)

    class_names = plotting.DEFAULT_NAMES
    num_classes = len(class_names)
    path_slices = {"control": slice(0, 6), "treatment": slice(6, 12),
                   "simulated": slice(12, 18)}
    baseline_avg = 1.0 / num_classes
    model_keys = list(results)

    with open(os.path.join(outdir, "mean_likelihood_per_config.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        header = ["configuration", "class"] + list(path_slices) + ["baseline"]
        w.writerow(header)
        for dhw, label in zip(CONFIG_ORDER, CONFIG_LABELS):
            per_path = {
                name: per_config_bootstrap_stats(
                    results, model_keys, num_classes, sl, n_bootstrap=1000,
                    ci=95, rng=np.random.default_rng(42))[0]
                for name, sl in path_slices.items()
            }
            idx = CONFIG_ORDER.index(dhw)
            for cls, cname in enumerate(class_names):
                row = [label, cname] + \
                      [f"{per_path[p][cls][idx]:.4f}" for p in path_slices] + \
                      [f"{baseline_avg:.4f}"]
                w.writerow(row)
    print(f"  wrote {os.path.join(outdir, 'mean_likelihood_per_config.csv')}")


def _save_results(results, path):
    import torch

    blob = {}
    for key, out in results.items():
        for i, hive in enumerate(out["hives"]):
            for k, v in hive.items():
                blob[f"{key}{i}_{k}"] = torch.as_tensor(v)
    torch.save(blob, path)
    print(f"  saved full results to {path}")


if __name__ == "__main__":
    make_figures_cli()
