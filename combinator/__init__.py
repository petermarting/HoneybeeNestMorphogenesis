"""combinator -- analysis and modeling of honeybee nest-content 4D atlas data.

Code, models, and data for Marting et al., "Morphogenesis, differentiation,
and robust multi-scale organization of the honeybee nest revealed by a novel
4D atlas", Current Biology (2026).

The package covers both halves of the paper:

* ``combinator.content`` -- the observational workflows for Figures 1–5,
  including seasonality, cell, comb, and nest organization, specialization,
  and the shuffling experiment. See RESULTS.md for reproduction coverage.
* the autoregressive masked-convolutional model and its data pipeline
  (training, evaluation, figure generation; paper Figure 6).

See README.md for setup and usage, and models/MANIFEST.md for the provenance
of each committed checkpoint.
"""

from .dataset import GridDataset, compute_class_weights, load_grids
from .eval import build_eval_hives, evaluate_spatial_classifier
from .labels import map_all_labels
from .make_figures import (
    evaluate_all_v2,
    make_figures,
    make_figures_cli,
    write_figure6_table,
    write_per_config_table,
)
from .models import (
    CANONICAL_GRID_SHAPE,
    SpatialClassifier,
    config_from_v2_name,
    from_config,
    load,
    load_any,
)
from .plotting import (
    CONFIG_LABELS,
    CONFIG_ORDER,
    DEFAULT_NAMES,
    per_config_bootstrap_stats,
    plot_config_by_treatment,
    plot_control_physical_simulated,
    plot_control_physical_simulated_full,
    plot_kernel_legend_subset,
    plot_likelihood_over_time,
    plot_likelihood_with_bootstrap,
    plot_model_scatter3d_simplified,
)
from .preprocessing import process_grids
from .registry import V2_MODELS, v2_config
from .results_io import load_results, save_results, verify_dir
from .train import train_all_v2, train_spatial_classifier

__version__ = "0.1.0"

__all__ = [
    "CANONICAL_GRID_SHAPE",
    "CONFIG_LABELS",
    "CONFIG_ORDER",
    "DEFAULT_NAMES",
    "V2_MODELS",
    "GridDataset",
    "SpatialClassifier",
    "build_eval_hives",
    "compute_class_weights",
    "config_from_v2_name",
    "evaluate_all_v2",
    "evaluate_spatial_classifier",
    "from_config",
    "load",
    "load_any",
    "load_grids",
    "load_results",
    "make_figures",
    "make_figures_cli",
    "map_all_labels",
    "per_config_bootstrap_stats",
    "plot_config_by_treatment",
    "plot_control_physical_simulated",
    "plot_control_physical_simulated_full",
    "plot_kernel_legend_subset",
    "plot_likelihood_over_time",
    "plot_likelihood_with_bootstrap",
    "plot_model_scatter3d_simplified",
    "process_grids",
    "save_results",
    "train_all_v2",
    "train_spatial_classifier",
    "v2_config",
    "verify_dir",
    "write_figure6_table",
    "write_per_config_table",
]
