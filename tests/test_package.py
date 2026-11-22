"""Package, loader, and figure-smoke tests (fast)."""

from __future__ import annotations

import numpy as np  # noqa: F401  (keeps seeded fixture's dep explicit)
import pytest

import combinator.content as content
from combinator.content import (celllevel, comblevel, nestlevel,
                                seasonality, shuffled, specialization)


def test_package_imports():
    for mod in (content.constants, content.data, content.util, content.make,
                content.seasonality, content.celllevel, content.comblevel,
                content.nestlevel, content.specialization, content.shuffled):
        assert mod is not None


def test_load_table_shapes_and_columns(tabs):
    assert tabs["df_combcounts"].shape == (4720, 332)
    df = tabs["df_mergedinsp"]
    assert df.shape == (4720, 334)
    for col in ("perc_bees_onframe", "nectar_honey", "perc_nectar_honey",
                "comb_perim_area_ratio", "frame_RO", "dispersion_metric",
                "dispersion_metric_nest"):
        assert col in df.columns, col
    # frame_RO: forward (physical) order for DD6/DD7/SH4, reversed elsewhere
    fwd = df[df.colony.isin(["DD6", "DD7", "SH4"])]
    assert (fwd.frame_RO == fwd.frame).all()
    rev = df[~df.colony.isin(["DD6", "DD7", "SH4"])]
    assert (rev.frame_RO == 11 - rev.frame).all()


def test_load_masks_shapes(masks, px_to_cm):
    contents = masks["Contents"]
    assert contents.dtype == object and len(contents) == 12
    crop = masks["ContentsCrop"][0][0][0]
    assert crop.shape == (42, 88)
    assert px_to_cm == pytest.approx(0.48863636363636365)


def test_seasonality_weekly_summary(tabs):
    w = seasonality.weekly_summary(tabs["df_mergedinsp"])
    assert w.true_week.min() == 1 and w.true_week.max() == 30
    ticks = seasonality.month_ticks(tabs["df_mergedinsp"])
    assert list(ticks.month) == ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
                                 'Oct', 'Nov']


def test_content_change_frames(tabs):
    df = shuffled.content_change_frames(tabs["df_mergedinsp"])
    assert "ttl_abs_change" in df and "perc_ttl_abs_change" in df
    # first week of each comb series has no change value
    assert df.loc[df.week == 1, "ttl_abs_change"].isna().all()


def test_figure_1_smoke(tabs, tmp_path):
    fig, ax = seasonality.figure_1_seasonality(
        tabs["df_mergedinsp"], out=str(tmp_path / "fig1.pdf"))
    assert len(ax) == 4 and fig is not None


def test_figure_2_network_smoke(tabs, tmp_path):
    celllevel.neighborhood_network(tabs["df_cellneib"],
                                   out=str(tmp_path / "fig2a.pdf"))
    celllevel.transition_network(tabs["df_cellfuture"],
                                 out=str(tmp_path / "fig2b.pdf"))


def test_figure_2_cluster_smoke(tabs, tmp_path):
    # n=200 smoke of the bootstrap mechanics; CI medians jitter with n
    boot = celllevel.bootstrap_cluster_metric(tabs["df_nearestdist"],
                                              n_bootstraps=200)
    assert len(boot) == 6
    celllevel.figure_2_cluster_by_content(
        tabs["df_nearestdist"], boot, out=str(tmp_path / "fig2d.pdf"))
    celllevel.figure_2_cluster_over_time(
        tabs["df_nearestdist"], out=str(tmp_path / "fig2e.pdf"))
    celllevel.figure_2_nest_metric_by_colony(
        tabs["df_nearestdist_nest_sum"], out=str(tmp_path / "fig2g.pdf"))


def test_figure_3_pca(tabs, contents, tmp_path):
    pca, loadings, df_pca, evr = comblevel.comb_pca(tabs["df_mergedinsp"])
    # PC1/PC2 explain 40% / 24% of variance (paper)
    assert evr[0] == pytest.approx(0.396, abs=0.005)
    assert evr[1] == pytest.approx(0.237, abs=0.005)
    comblevel.figure_3_comb_pca(df_pca, loadings, evr, contents,
                                out=str(tmp_path / "fig3a.pdf"))


def test_figure_3_vertical(tabs, masks, tmp_path):
    df = comblevel.vertical_distribution_frames(masks["Contents"])
    assert len(df) > 5_000_000
    comblevel.figure_3_vertical_distribution(
        df, masks["ContentsCrop"], out=str(tmp_path / "fig3b.pdf"))


def test_figure_3_ridgelines_smoke(tabs, tmp_path):
    nestlevel.figure_3_combtypes_across_nest(
        tabs["df_mergedinsp"], out=str(tmp_path / "fig3c.pdf"))
    nestlevel.figure_3_contents_across_nest(
        tabs["df_mergedinsp"], out=str(tmp_path / "fig3d.pdf"))


def test_figure_5_example_ridgelines_smoke(tabs, tmp_path):
    shuffled.figure_5_example_ridgelines(
        tabs["df_mergedinsp"], out=str(tmp_path / "fig5e.pdf"))


@pytest.mark.slow
def test_figure_4b_sequences(contents, tmp_path):
    specialization.figure_4_cell_sequences(
        contents, out=str(tmp_path / "fig4b.pdf"))
