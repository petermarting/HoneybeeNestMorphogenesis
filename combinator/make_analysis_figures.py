"""Regenerate supported observational figure panels for Figures 1–5.

Run from the repository root.

    uv run python -m combinator.make_analysis_figures --outdir figures_out/analysis

The workflow uses the released comb masks and tables and writes one PDF
per panel group. Pass ``--figs 1,3`` to render a subset. The bootstrap seed
defaults to 42. Pass ``--seed -1`` to use unseeded draws.

Figures 3B–D are generated from the available data. See RESULTS.md for the
published statistics and panels that need additional inputs or code.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .content import (celllevel, comblevel, data, nestlevel, seasonality,
                      shuffled, specialization)


def _figure_figs1(tabs, masks, out):
    seasonality.figure_1_seasonality(tabs["df_mergedinsp"],
                                     out=os.path.join(out, "fig1_seasonality.pdf"))


def _figure_figs2(tabs, masks, out):
    celllevel.neighborhood_network(tabs["df_cellneib"],
                                   out=os.path.join(out, "fig2a_neighbourhood_network.pdf"))
    celllevel.transition_network(tabs["df_cellfuture"],
                                 out=os.path.join(out, "fig2b_transition_network.pdf"))
    boot = celllevel.bootstrap_cluster_metric(tabs["df_nearestdist"])
    celllevel.figure_2_cluster_by_content(
        tabs["df_nearestdist"], boot,
        out=os.path.join(out, "fig2d_comb_cluster_metric.pdf"))
    celllevel.figure_2_cluster_over_time(
        tabs["df_nearestdist"],
        out=os.path.join(out, "fig2e_comb_cluster_over_time.pdf"))
    boot_nest = celllevel.bootstrap_cluster_metric(tabs["df_nearestdist_nest"])
    celllevel.figure_2_cluster_by_content(
        tabs["df_nearestdist_nest"], boot_nest, jitter=.2, alpha=.5,
        ylabel='Nest-level cluster metric',
        out=os.path.join(out, "fig2f_nest_cluster_metric.pdf"))
    celllevel.figure_2_nest_metric_by_colony(
        tabs["df_nearestdist_nest_sum"],
        out=os.path.join(out, "fig2g_nest_metric_by_colony.pdf"))


def _figure_figs3(tabs, masks, out):
    pca, loadings, df_pca, evr = comblevel.comb_pca(tabs["df_mergedinsp"])
    comblevel.figure_3_comb_pca(df_pca, loadings, evr, masks["Contents"],
                                out=os.path.join(out, "fig3a_comb_pca.pdf"))
    df_coords = comblevel.vertical_distribution_frames(masks["Contents"])
    comblevel.figure_3_vertical_distribution(
        df_coords, masks["ContentsCrop"],
        out=os.path.join(out, "fig3b_vertical_distribution.pdf"))
    nestlevel.figure_3_combtypes_across_nest(
        tabs["df_mergedinsp"], out=os.path.join(out, "fig3c_combtypes_across_nest.pdf"))
    nestlevel.figure_3_contents_across_nest(
        tabs["df_mergedinsp"], out=os.path.join(out, "fig3d_contents_across_nest.pdf"))


def _figure_figs4(tabs, masks, out):
    px = data.px_to_cm(masks["ContentsCrop"])
    specialization.figure_4_cell_sequences(
        masks["Contents"], out=os.path.join(out, "fig4b_cell_sequences.pdf"))
    strat = specialization.stratification_sequences(masks["Contents"])
    specialization.figure_4_stratification(
        strat, masks["Contents"], px,
        out=os.path.join(out, "fig4d_stratification.pdf"))
    dcp = specialization.cell_specialization_comparison(masks["Contents"])
    specialization.figure_4_specialization_vs_randomized(
        dcp, out=os.path.join(out, "fig4e_specialization_vs_randomized.pdf"))
    comb_sp = specialization.comb_specialization(masks["Contents"])
    specialization.figure_4_comb_specialization(
        comb_sp, out=os.path.join(out, "fig4f_comb_specialization.pdf"))


def _figure_figs5(tabs, masks, out):
    med, err = shuffled.life_stage_bootstraps(tabs["df_mergedinsp"])
    shuffled.figure_5_life_stages(
        tabs["df_mergedinsp"], med, err,
        out=os.path.join(out, "fig5b_life_stages.pdf"))
    df_broodflow = shuffled.brood_flow(masks["Contents"])
    df_sum = shuffled.brood_flow_summary(shuffled.brood_outcomes(df_broodflow))
    shuffled.figure_5_brood_survival(
        df_sum, out=os.path.join(out, "fig5c_brood_survival.pdf"))
    df_changes = shuffled.content_change_frames(tabs["df_mergedinsp"])
    med2, err2 = shuffled.content_change_bootstrap(df_changes)
    shuffled.figure_5_content_fluctuations(
        df_changes, err2,
        out=os.path.join(out, "fig5d_content_fluctuations.pdf"))
    shuffled.figure_5_example_ridgelines(
        tabs["df_mergedinsp"],
        out=os.path.join(out, "fig5e_ridgelines_dd2_sh2.pdf"))
    comb_sp = shuffled.comb_specialization_all(masks["Contents"])
    shuffled.figure_5_comb_specialization(
        comb_sp, out=os.path.join(out, "fig5fg_comb_specialization.pdf"))


_FIG_BUILDERS = {1: _figure_figs1, 2: _figure_figs2, 3: _figure_figs3,
                 4: _figure_figs4, 5: _figure_figs5}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--figs", default="all",
                    help="comma-separated figure numbers (1-5) or 'all'")
    ap.add_argument("--root", default=None,
                    help="directory holding df_*.csv and CombMasks_*.npy")
    ap.add_argument("--outdir", default=os.path.join("figures", "analysis"))
    ap.add_argument("--seed", type=int, default=42,
                    help="numpy seed for bootstraps (-1 = unseeded)")
    args = ap.parse_args()

    if args.seed >= 0:
        np.random.seed(args.seed)

    figs = ({1, 2, 3, 4, 5} if args.figs.strip() == "all"
            else {int(x) for x in args.figs.split(",")})
    unknown = figs - set(_FIG_BUILDERS)
    if unknown:
        raise SystemExit(f"unknown figure(s): {sorted(unknown)}")

    os.makedirs(args.outdir, exist_ok=True)
    tabs = masks = None
    if {1, 2, 3, 5} & figs:
        tabs = data.load_tables(args.root)
    if {3, 4, 5} & figs:
        masks = data.load_masks(args.root)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for n in sorted(figs):
        print(f"Figure {n} ...", flush=True)
        _FIG_BUILDERS[n](tabs, masks, args.outdir)
        plt.close("all")


if __name__ == "__main__":
    main()
