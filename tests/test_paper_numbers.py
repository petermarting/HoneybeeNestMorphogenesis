"""Check selected quantitative results for Figures 1–5.

The default suite uses released masks and tables. Tests marked ``slow``
cover longer bootstrap, figure, and table checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from combinator.content import celllevel, make, seasonality, shuffled, specialization
from conftest import assert_stats

T95 = 2.4469  # t(5, 0.975): 95% CI factor over n=6 colonies


def mci(vals):
    """Mean ± 95% CI (t-based, n=6) — the paper's colony-level summary."""
    vals = np.asarray(vals, dtype=float)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals)) * T95


# --------------------------------------------------------------------------
# Figure 1: week-10 nest composition ("brood 32.7% ± 7.7% ...")
# --------------------------------------------------------------------------

def test_seasonality_week10_paper_numbers(tabs):
    w = seasonality.weekly_summary(tabs["df_mergedinsp"])
    w = w[w.true_week == 10].copy()
    w["nectar_honey"] = w["nectar"] + w["honey"]
    for vals, mean, ci in [
        (w.allbrood / w.totalcomb, 0.327, 0.077),
        (w.pollen / w.totalcomb, 0.085, 0.055),
        (w.nectar_honey / w.totalcomb, 0.145, 0.032),
        (w.empty_comb / w.totalcomb, 0.443, 0.085),
        (w.totaldronecomb / (w.totalworkercomb + w.totaldronecomb),
         0.117, 0.048),
    ]:
        obs = mci(vals)
        assert abs(obs[0] - mean) <= 0.0015, (mean, obs)
        assert abs(obs[1] - ci) <= 0.0015, (ci, obs)


# --------------------------------------------------------------------------
# Figure 2: cluster metric ("0.803 ± 0.145 ... 7,746 of 7,765 ...")
# --------------------------------------------------------------------------

def test_comb_cluster_metric_paper_numbers(tabs):
    s = celllevel.cluster_metric_stats(tabs["df_nearestdist"])
    assert_stats(s["first10_overall"], 0.803, 0.145)
    assert_stats(s["first10_honey"], 0.959, 0.066)
    assert_stats(s["week1"], 0.912, 0.077)
    assert_stats(s["week4"], 0.801, 0.145)
    assert_stats(s["after_week4"], 0.785, 0.146)
    assert s["n_obs"] == 7765
    assert s["n_significant"] == 7746


def test_nest_cluster_metric_paper_numbers(tabs):
    s = celllevel.nest_cluster_metric_stats(tabs["df_nearestdist_nest"])
    assert s["first10_mean"] == pytest.approx(0.849, abs=0.002)
    assert s["first10_sd"] == pytest.approx(0.097, abs=0.003)
    assert s["n_obs"] == 644
    assert s["n_significant"] == 644


# --------------------------------------------------------------------------
# Figure 4: cell specialization ("60.9% ± 3.9% of cells specialized ...")
# --------------------------------------------------------------------------

def test_cell_specialist_thresholds(contents):
    cell_sp = specialization.cell_specialization(contents)
    # thresholds printed by the notebook's pipeline
    assert np.allclose(cell_sp.prop_thres,
                       [74.72, 68.48, 24.96, 35.23], atol=0.01)
    # pooled proportions: generalists 38.73%, empty 20.05%, brood 18.97%,
    # pollen 11.35%, nectar 10.9%
    assert np.allclose(cell_sp.class_props,
                       [38.73, 20.05, 18.97, 11.35, 10.90], atol=0.05)


def test_cell_specialist_share(contents):
    # per-colony share of specialized cells under the shared thresholds
    dcp = specialization.cell_specialization_comparison(contents)
    obs = dcp[dcp.treatment == "observed"]
    per_colony = obs.groupby("colony").apply(
        lambda g: g.loc[g.class_ids != 0, "class_props"].sum(),
        include_groups=False)
    assert per_colony.mean() == pytest.approx(60.9, abs=0.05)
    assert per_colony.std() == pytest.approx(3.9, abs=0.05)  # sample SD
    # per-class spectrum quoted in the paper (mean ± SD over colonies)
    for name, mean, sd in (("generalists", 39.1, 3.9),
                           ("empty specialists", 20.4, 5.9),
                           ("brood specialists", 18.9, 7.7),
                           ("pollen specialists", 11.0, 4.9),
                           ("nectar specialists", 10.6, 5.0)):
        v = obs.loc[obs.specialization == name, "class_props"]
        assert v.mean() == pytest.approx(mean, abs=0.05)
        assert v.std() == pytest.approx(sd, abs=0.05)


# --------------------------------------------------------------------------
# Figure 5: brood failure ("control = 2,863 ± 488 ... 88.9% higher ...")
# --------------------------------------------------------------------------

def test_brood_failure_counts(contents):
    df = shuffled.brood_outcomes(shuffled.brood_flow(contents))
    got = {}
    for treat in ("DD", "SH"):
        for src in ("eggs", "larvae"):
            sub = df[(df.treatment == treat) & (df.source_name == src)
                     & (df.target_outcome == "failed")]
            per_colony = sub.groupby("colony").size()
            got[(treat, src)] = (per_colony.mean(), per_colony.std())
    # paper: per-colony failed-cell totals over the 9 weekly transitions
    # (8,224.5 reported as 8,225 by conventional rounding)
    for key, mean, sd in ((("DD", "eggs"), 2863, 488),
                          (("DD", "larvae"), 7260, 1003),
                          (("SH", "eggs"), 5407, 1966),
                          (("SH", "larvae"), 8225, 2256)):
        assert abs(got[key][0] - mean) <= 0.5, key
        assert abs(got[key][1] - sd) <= 0.5, key
    # "88.9% higher rate ... for eggs ... 13.3% ... for larvae"
    for src, pct in (("eggs", 88.9), ("larvae", 13.3)):
        ratio = (got[("SH", src)][0] / got[("DD", src)][0] - 1) * 100
        assert round(ratio, 1) == pct


def test_comb_specialization_shuffle_comparison(contents):
    comb_sp = shuffled.comb_specialization_all(contents)
    ids = np.asarray(comb_sp.comb_ids)
    spec = np.isin(ids, [2, 5, 6, 7])  # non-generalist classes
    # the DD6 frame-10 comb was never built (blanked in the figure)
    dd = spec[:6].sum() - spec[4, 9]
    sh = spec[6:].sum()
    assert dd == 34
    assert sh == 15
    assert round((1 - sh / dd) * 100, 1) == 55.9


# --------------------------------------------------------------------------
# Make-notebook generators reproduce the committed tables
# --------------------------------------------------------------------------

def test_make_cell_neighbour_subset(contents, tabs):
    # Compare DD1 weeks 1-2 with the released df_cellneib rows.
    sub = np.empty(1, dtype=object)
    sub[0] = contents[0][:2]
    df = make.cell_neighbour_table(sub)
    ref = tabs["df_cellneib"]
    ref = ref[(ref.colony == "DD1") & (ref.week <= 2)].reset_index(drop=True)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), ref,
                                  check_exact=False, rtol=1e-9)


def test_make_cell_future_subset(contents, tabs):
    sub = np.empty(1, dtype=object)
    sub[0] = contents[0][:3]
    df = make.cell_future_table(sub)
    df = df[df.week <= 2].reset_index(drop=True)
    ref = tabs["df_cellfuture"]
    ref = ref[(ref.colony == "DD1") & (ref.week <= 2)].reset_index(drop=True)
    pd.testing.assert_frame_equal(df, ref, check_exact=False, rtol=1e-9)


# --------------------------------------------------------------------------
# Seeded bootstrap medians vs paper (slow: n=4000 across contents/traits)
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_life_stage_bootstrap_medians(tabs):
    np.random.seed(42)
    med, _err = shuffled.life_stage_bootstraps(tabs["df_mergedinsp"])
    dd, sh = med[0], med[1]
    # paper (hierarchical bootstrapping): control 1,825 ± 162 eggs /
    # 5,109 ± 404 larvae / 10,133 ± 1,259 pupae / 13,816 ± 1,120 adults;
    # shuffled 2,517 ± 167 / 5,831 ± 471 / 10,992 ± 1,397 / 13,806 ± 1,033
    for obs, exp in ((dd[0], 1825), (sh[0], 2517), (dd[1], 5109),
                     (sh[1], 5831), (dd[2], 10133), (sh[2], 10992),
                     (dd[3], 13816), (sh[3], 13806)):
        assert obs == pytest.approx(exp, abs=200)  # unseeded-vs-seeded jitter
    # "mean difference ... 686.1 ± 510" more eggs in shuffled
    assert (sh[0] - dd[0]) == pytest.approx(686, abs=400)


@pytest.mark.slow
def test_content_fluctuation_bootstrap(tabs):
    np.random.seed(42)
    df = shuffled.content_change_frames(tabs["df_mergedinsp"])
    med, _err = shuffled.content_change_bootstrap(df)
    # paper: 32.7% (control) vs 46.9% (shuffled), difference 14 ± 5.1
    assert med[0] == pytest.approx(32.7, abs=1.5)
    assert med[1] == pytest.approx(46.9, abs=1.5)
    assert (med[1] - med[0]) == pytest.approx(14.2, abs=5.1)


@pytest.mark.slow
def test_nearest_neighbour_table_full_reproduction(tabs, masks):
    """Rebuild df_nearestdist from the masks with the pipeline seed.

    Compare deterministic fields and Monte Carlo statistics using their
    respective numeric tolerances.
    """
    df, _sum = make.nearest_neighbour_table(masks["ContentsCrop"], seed=42)
    ref = tabs["df_nearestdist"]
    assert len(df) == len(ref)
    for col in ("count", "combcount", "nearestneibavg", "content_density",
                "week", "frame", "contentid"):
        assert np.allclose(df[col].values, ref[col].values, rtol=1e-9), col
    # n=100 Monte-Carlo p-values quantize to ~1/101 steps; two independent
    # 100-sample streams can legitimately differ by ~0.1-0.2.
    assert np.allclose(df["dispersion_metric"].values,
                       ref["dispersion_metric"].values, atol=0.06)
    assert np.allclose(df["p_value"].values, ref["p_value"].values, atol=0.2)
