"""Cell-level organization (paper Figure 2).

Derived from the paper's ``CombContent_Analysis.ipynb``.

Panels (legend, Figure 2): (A) local-neighborhood composition network;
(B) week-to-week content-transition network (spring growth); (C) a
schematic explaining the cluster metric — drawn, not computed, so no code
reproduces it (see ``RESULTS.md``); (D) comb-level cluster metric by
content, first 10 weeks (bootstrapped median ± 95% CI); (E) comb-level
cluster metric over the whole study; (F) nest-level cluster metric by
content, first 10 weeks; (G) weighted-average nest-level cluster metric
per colony over time.

The cluster metric itself (normalized observed vs randomized same-content
nearest-neighbour proximity) was computed offline with a pylandstats-based
pipeline and shipped in ``df_nearestdist*.csv``; the
Monte-Carlo p-values in those tables are likewise committed, not
recomputed here.  See ``RESULTS.md``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from . import util
from .constants import colony_deadout_week, content_names, contenthex

#: Brood + resource contents shown in the Figure 2 network and cluster
#: panels (``content_names[3:9]``: eggs, larvae, pupae, pollen, nectar,
#: honey).
CONTENT_ORDER = content_names[3:9]


def _content_colors() -> list[str]:
    """Palette for ``CONTENT_ORDER``: content hexes + lightgrey for honey."""
    colors = contenthex[3:8]
    colors.append('lightgrey')
    return colors


# ---------------------------------------------------------------------------
# Networks (Figures 2A, 2B)
# ---------------------------------------------------------------------------

def _network_figure(data: pd.DataFrame, dest_col: str, metric: str,
                    weight_div: float) -> tuple:
    """Circular directed network of focal-content -> destination-content.

    Shared engine for Figure 2A (neighbour composition: ``dest_col``
    ``'neibcontent'``, ``metric`` ``'neibperc'``) and Figure 2B (temporal
    transitions: ``dest_col`` ``'nextself'``, ``metric`` ``'nextperc'``):
    one node per content; curved directed edges weighted by the mean
    proportion of the destination content (arrows indicate focal ->
    destination and are *not* causal); self-loop arcs (``draw_arrow_loop``)
    sized by the focal->self proportion.  Edge widths use
    ``abs(metric)/weight_div`` (÷2 neighbours, ÷3 transitions), self-loop
    widths always ÷2 — exactly as in the notebook.
    """
    g = nx.MultiDiGraph()
    colors = contenthex[2:8]
    colors.append('lightgrey')

    for i, content in enumerate(CONTENT_ORDER):
        g.add_node(content, size=2000, node_color=colors[i + 1])
        for other in CONTENT_ORDER:
            if content == other:
                continue
            df = data[(data['focalcontent'] == content)
                      & (data[dest_col] == other)]
            g.add_edge(content, other,
                       weight=abs(df[metric]).iloc[0] / weight_div,
                       color=colors[i + 1])

    c = nx.get_edge_attributes(g, 'color').values()
    weights = nx.get_edge_attributes(g, 'weight').values()
    nc = nx.get_node_attributes(g, 'node_color').values()
    sizes = nx.get_node_attributes(g, 'size').values()

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.circular_layout(g)
    n = nx.draw_networkx_nodes(g, pos, ax=ax, node_size=list(sizes),
                               node_color=list(nc))
    nx.draw_networkx_labels(g, pos, ax=ax)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=c, width=list(weights),
                           connectionstyle='arc3, rad = 0.1', arrows=True,
                           arrowsize=20, alpha=1)
    n.set_zorder(0)
    ax.axis('off')

    # self loops
    radius = .7
    theta2_ = 265
    angles = 210 + np.arange(0, len(CONTENT_ORDER)) * (360 / len(CONTENT_ORDER))
    for i, content in enumerate(CONTENT_ORDER):
        df = data[(data['focalcontent'] == content)
                  & (data[dest_col] == content)]
        x = pos[content][0]
        y = pos[content][1]
        centX, centY = util.extrapolate_point(0, 0, x, y, .4)
        util.draw_arrow_loop(ax, radius, abs(df[metric]).iloc[0] / 2, centX,
                             centY, angles[i], theta2_, colors[i + 1])
    return fig, ax


def neighborhood_network(df_cellneib: pd.DataFrame, treatment: str = 'DD',
                         maxweek: int = 11, out: str | None = None):
    """Figure 2A — immediate-neighbour composition network.

    Mean neighbourhood composition over all focal cells (one colony-week-
    frame-side row each) of ``treatment`` up to (not including) week
    ``maxweek``; edge width = proportion of destination-content neighbours
    around focal cells of the origin content.
    """
    data = df_cellneib[(df_cellneib.week < maxweek)
                       & (df_cellneib.treatment == treatment)]
    data = data.groupby(by=['focalcontent', 'neibcontent'],
                        as_index=False).mean(numeric_only=True)
    fig, ax = _network_figure(data, 'neibcontent', 'neibperc', weight_div=2)
    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax


def transition_network(df_cellfuture: pd.DataFrame, treatment: str = 'DD',
                       maxweek: int = 11, out: str | None = None):
    """Figure 2B — week-to-week content transition network.

    Spring-growth period (``week < maxweek``): edge width = proportion of
    cells of the origin content that became the destination content one
    week later.
    """
    data = df_cellfuture[(df_cellfuture.week < maxweek)
                         & (df_cellfuture.treatment == treatment)]
    data = data.groupby(by=['focalcontent', 'nextself'],
                        as_index=False).mean(numeric_only=True)
    fig, ax = _network_figure(data, 'nextself', 'nextperc', weight_div=3)
    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax


# ---------------------------------------------------------------------------
# Cluster metric (Figures 2D–G)
# ---------------------------------------------------------------------------

def bootstrap_cluster_metric(df_nearestdist: pd.DataFrame,
                             treatment: str = 'DD', maxweek: int = 11,
                             n_bootstraps: int = 4000) -> list[dict]:
    """Hierarchical bootstrap of the cluster metric, per content (Fig 2D/F).

    Colony-level bootstrap (hierarchy ``['colony']``) of
    ``dispersion_metric`` for each of ``CONTENT_ORDER`` over the first
    ``maxweek - 1`` weeks, returning ``bootstrap_summary_stats`` dicts
    (median, lower/upper bounds, errors) — the notebook prints
    ``content: median (lower - upper)`` per content.

    NOTE: bootstrapping uses the global ``numpy.random`` stream; seed it
    before calling for reproducibility (paper values are unseeded, n=4000).
    """
    data = df_nearestdist[(df_nearestdist['treatment'] == treatment)
                          & (df_nearestdist['week'] < maxweek)]
    hierarchy_levels = ['colony']
    target = 'dispersion_metric'
    boot_sums = []
    for content in CONTENT_ORDER:
        data_sub = data[data['content'] == content]
        data_sub = data_sub[hierarchy_levels + [target]]
        bootstrap_res = util.hierarchical_bootstrap(
            data_sub, hierarchy_levels, n_bootstraps=n_bootstraps)
        summary = util.bootstrap_summary_stats(bootstrap_res[target].values)
        boot_sums.append(summary)
        print(content + ': ' + str(summary['median']) + ' ('
              + str(summary['lower_bounds']) + ' - '
              + str(summary['upper_bounds']) + ')')
    return boot_sums


def figure_2_cluster_by_content(df_nearestdist: pd.DataFrame,
                                boot_sums: list[dict],
                                treatment: str = 'DD', maxweek: int = 11,
                                ylabel: str = 'Cluster metric',
                                jitter: float = .3, alpha: float = .1,
                                out: str | None = None):
    """Figures 2D/2F — stripplot of all observations + bootstrap CI points.

    Call ``bootstrap_cluster_metric`` on the same (already-filtered)
    subset to obtain ``boot_sums``: 2D uses the comb table
    (``df_nearestdist``) with defaults, 2F the nest table
    (``df_nearestdist_nest``) with ``jitter=.2, alpha=.5,
    ylabel='Nest-level cluster metric'``.
    """
    data = df_nearestdist[(df_nearestdist['treatment'] == treatment)
                          & (df_nearestdist['week'] < maxweek)]
    colors = _content_colors()

    fig, ax0 = plt.subplots(figsize=(2.7, 4))
    sns.stripplot(ax=ax0, data=data, y='dispersion_metric', x='content',
                  hue='content', palette=colors, legend=False, jitter=jitter,
                  alpha=alpha, order=CONTENT_ORDER)
    medians = [sublist['median'] for sublist in boot_sums]
    errors = [sublist['errors'] for sublist in boot_sums]
    ax0.errorbar(x=np.arange(len(CONTENT_ORDER)), y=medians,
                 yerr=np.array(errors).T, fmt='o', capsize=3, color='k',
                 zorder=10000)
    ax0.set_xlabel('Content')
    ax0.set_ylabel(ylabel)
    ax0.set_xticks(np.arange(len(CONTENT_ORDER)), labels=CONTENT_ORDER,
                   rotation=70)
    ax0.plot([-.5, 5.5], [1, 1], c='grey', ls='--')
    ax0.plot([-.5, 5.5], [0, 0], c='grey', ls='--')
    ax0.text(2.5, 1.05, 'maximum clustering', horizontalalignment='center')
    ax0.text(2.5, .05, 'randomly distributed', horizontalalignment='center')
    ax0.plot([2.5, 2.5], [.03, 0], c='grey')
    ax0.plot([2.5, 2.5], [1.03, 1], c='grey')
    sns.despine()
    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax0


def figure_2_cluster_over_time(df_nearestdist: pd.DataFrame,
                               treatment: str = 'DD',
                               out: str | None = None):
    """Figure 2E — comb-level cluster metric per content over all weeks.

    Dead-colony weeks excluded.  NOTE: the paper legend states honey
    cells were absent from all nests weeks 12–25, so the honey line has
    a gap there (seaborn simply skips missing groups).
    """
    data = df_nearestdist[(df_nearestdist['treatment'] == treatment)
                          & (~df_nearestdist['true_week']
                             .isin(colony_deadout_week))]
    colors = _content_colors()

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.pointplot(ax=ax, data=data, x='true_week', y='dispersion_metric',
                  hue='content', palette=colors, dodge=True,
                  hue_order=CONTENT_ORDER, native_scale=True)
    sns.despine()
    ax.set_xlabel('Time (weeks)')
    ax.set_ylabel('Cluster metric')
    ax.legend(frameon=False)
    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax


def figure_2_nest_metric_by_colony(df_nearestdist_nest_sum: pd.DataFrame,
                                   treatment: str = 'DD',
                                   out: str | None = None):
    """Figure 2G — weighted-average nest-level cluster metric per colony.

    One line per colony over all (non-deadout) weeks; colonies are
    especially synchronized early on.
    """
    data = df_nearestdist_nest_sum[
        (df_nearestdist_nest_sum['treatment'] == treatment)
        & (~df_nearestdist_nest_sum['true_week'].isin(colony_deadout_week))]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.pointplot(ax=ax, data=data, x='true_week', y='dispersion_metric',
                  hue='colony', palette='magma', native_scale=True)
    sns.despine()
    ax.set_xlabel('Time (weeks)')
    ax.set_ylabel('Nest-level cluster metric (weighted avg)')
    ax.legend(frameon=False, loc='lower right', bbox_to_anchor=(.65, 0))
    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax


# ---------------------------------------------------------------------------
# Paper-number summary (for RESULTS.md / tests)
# ---------------------------------------------------------------------------

def cluster_metric_stats(df_nearestdist: pd.DataFrame, treatment: str = 'DD',
                         ) -> dict:
    """Deterministic cluster-metric statistics quoted in the paper text.

    Reproduces (mean ± SD over colony-week-content observations unless
    noted): overall first-10-weeks value ("0.803 ± 0.145"), honey
    ("0.959 ± 0.066"), week 1 ("0.912 ± 0.077"), week 4 ("0.801 ±
    0.145"), after week 4 ("0.785 ± 0.146"), and the Monte-Carlo
    significance counts ("p < 0.05 for 7,746 of 7,765 observations";
    nest table: "644 of 644", "0.849 ± 0.097").
    """
    first10 = df_nearestdist[(df_nearestdist['treatment'] == treatment)
                             & (df_nearestdist['week'] < 11)]

    def ms(vals):
        vals = np.asarray(vals, dtype=float)
        return {'mean': float(np.mean(vals)), 'sd': float(np.std(vals))}

    week4 = first10[first10['true_week'] == 4]
    after4 = first10[first10['true_week'] > 4]
    dd = df_nearestdist[df_nearestdist['treatment'] == treatment]
    sig = dd[dd['p_value'] < 0.05]
    return {
        'first10_overall': ms(first10['dispersion_metric']),
        'first10_honey': ms(
            first10[first10['content'] == 'honey']['dispersion_metric']),
        'week1': ms(first10[first10['true_week'] == 1]['dispersion_metric']),
        'week4': ms(week4['dispersion_metric']),
        'after_week4': ms(after4['dispersion_metric']),
        'n_obs': int(len(dd)),
        'n_significant': int(len(sig)),
    }


def nest_cluster_metric_stats(df_nearestdist_nest: pd.DataFrame,
                              treatment: str = 'DD') -> dict:
    """Deterministic nest-level statistics quoted in the paper text.

    "p < 0.05 for 644 of 644 observations, cluster metric = 0.849 ±
    0.097" — the mean ± SD is over the first-10-weeks nest observations
    (matching Figures 2F/2G), the significance count over all nest
    observations of the treatment.
    """
    first10 = df_nearestdist_nest[
        (df_nearestdist_nest['treatment'] == treatment)
        & (df_nearestdist_nest['week'] < 11)]
    all_obs = df_nearestdist_nest[
        df_nearestdist_nest['treatment'] == treatment]
    return {
        'first10_mean': float(np.mean(first10['dispersion_metric'])),
        'first10_sd': float(np.std(first10['dispersion_metric'])),
        'n_obs': int(len(all_obs)),
        'n_significant': int((all_obs['p_value'] < 0.05).sum()),
    }
