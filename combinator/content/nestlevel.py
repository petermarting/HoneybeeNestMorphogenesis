"""Nest-level organization (paper Figure 3C, 3D).

NOTE: the polarization percentages quoted in the paper text for these
panels ("89.9% ± 8.3% of the nest's pollen was in the combs on one
half ... 82.5% ± 32.1% ...") have no released code defining the half
split; no candidate definition reproduces them from the committed
tables (closest: ~91% ± 11 pollen on the pollen-heavy occupied half).
They are treated like the Figure 3B KS p-values — computed offline,
documented in ``RESULTS.md``.

Implements the ``# Nest Level`` section (cells 47–48) of the original
``CombContent_Analysis.ipynb``.

Panels (legend, Figure 3C/D): (C) mean distribution of worker and drone
comb across the combs of all nests (x axis = reoriented comb position,
stacked horizontal planes = weeks; all DD colonies pooled); (D)
distribution of simplified contents (any brood, pollen, nectar/honey)
for each colony separately.  Comb positions are re-oriented so frame 1
is against the wall closest to where the nest was initiated, and
amounts are normalized to the maximum per type per subplot.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from .constants import colony_names


def _ridgeline(df: pd.DataFrame, ax: plt.Axes, x_col: str,
               series: list[tuple[str, str, float]], yoffset: float,
               lw: float, alpha: float = 1.0, edge_min: float = 100.0) -> None:
    """Stacked week-planes ridgeline on one axis (shared engine of 3C/3D).

    One plane per week: filled curves of each ``series`` entry
    ``(column, colour, max_value)`` over reoriented comb position, each
    divided by its subplot-wide maximum ``max_value``; weeks that are
    entirely empty are skipped and the active nest-edge combs
    (``totalcomb`` > ``edge_min``) define the span of each plane.  Grey
    baselines separate the planes.
    """
    weeks = sorted(df['week'].unique())
    for w in weeks:
        dfw = df[df['week'] == w]
        if len(dfw) == 0:
            continue
        dfw = dfw.sort_values(x_col)

        # active nest edges
        comb_last = 0
        nest_edge_L = None
        for i in range(len(dfw)):
            comb = dfw['totalcomb'].iloc[i]
            if (comb > comb_last) & (comb_last < edge_min):
                nest_edge_L = dfw[x_col].iloc[i]
                break
        comb_last = 0
        nest_edge_R = None
        for i in range(len(dfw) - 1, -1, -1):
            comb = dfw['totalcomb'].iloc[i]
            if (comb > 100) & (comb_last < edge_min):
                nest_edge_R = dfw[x_col].iloc[i]
                break
        if nest_edge_L is None or nest_edge_R is None:
            continue
        dfw = dfw[(dfw[x_col] >= nest_edge_L)
                  & (dfw[x_col] <= nest_edge_R)]
        x = np.array(dfw[x_col])

        ax.plot([np.min(x), np.max(x)], [0 - yoffset * w] * 2, c='grey',
                lw=.5, zorder=1)
        for j, (col, color, colmax) in enumerate(series):
            y = np.array(dfw[col]) / colmax - yoffset * w
            zorder = w * 5 + j
            ax.fill_between(x, y, 0 - yoffset * w, alpha=alpha, color=color,
                            zorder=zorder, interpolate=False)
            ax.plot(x, y, c="w", lw=lw, zorder=zorder - 1)
    ax.set_xlim(np.min(df[x_col]) - np.max(df[x_col]) * .1)


def figure_3_contents_across_nest(df_mergedinsp: pd.DataFrame,
                                  maxweek: int = 11, out: str | None = None):
    """Figure 3D — ridgeline of simplified contents across the nest per colony.

    One subplot per unshuffled colony (top row DD1/DD2/DD4, bottom
    DD5/DD6/DD7); frame position reoriented per colony (DD6/DD7 keep
    ``frame`` order, the rest are reversed), contents normalized per
    colony.  Nectar accumulates opposite the pollen bias.
    """
    fig, ax = plt.subplots(3, 2, figsize=(6.6, 15))
    ax = ax.reshape(-1)
    yoffset = .7
    lw = 2.5

    for c in range(6):
        data = df_mergedinsp[(df_mergedinsp['colony'] == colony_names[c])
                             & (df_mergedinsp['week'] < maxweek + 1)]
        data = data[['colony', 'week', 'frame', 'pollen', 'nectar_honey',
                     'allbrood', 'totalcomb']].reset_index(
            drop=True).groupby(['colony', 'week', 'frame'],
                               as_index=False).mean()

        data['frame_RO'] = 11 - data['frame']
        if colony_names[c] in ('DD6', 'DD7'):
            data['frame_RO'] = 11 - data['frame_RO']

        colormap = 'viridis'
        Nmax = np.max(data.week) + 2
        Nmin = np.min(data.week) - 2
        Nlen = int(Nmax - Nmin)
        cmap = mpl.colormaps.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, Nlen))
        series = [('allbrood', colors[7], np.max(data.allbrood)),
                  ('nectar_honey', colors[4], np.max(data.nectar_honey)),
                  ('pollen', colors[10], np.max(data.pollen))]
        _ridgeline(data, ax[c], 'frame_RO', series, yoffset, lw)

        if c == 3:
            legend_elements = [
                Line2D([0], [0], marker='o', color=colors[10], label='pollen',
                       markersize=10, lw=0),
                Line2D([0], [0], marker='o', color=colors[7],
                       label='any brood', markersize=10, lw=0),
                Line2D([0], [0], marker='o', color=colors[4],
                       label='nectar\n& honey', markersize=10, lw=0),
            ]
            ax[c].legend(handles=legend_elements, loc='upper right',
                         bbox_to_anchor=(1.1, 1), edgecolor='w', facecolor='w',
                         framealpha=.7).set_zorder(1000)
        if c in (0, 2, 4):
            for w in range(1, int(data.week.max()) + 1):
                label = 'week\n%s' % w if w == 1 else str(w)
                ax[c].text(.2, 0 - yoffset * w, label, zorder=1000,
                           ha='center')
        if c > 2:
            ax[c].set_xlabel("Comb position across nest")
            ax[c].spines[['left', 'right', 'top']].set_visible(False)
            ax[c].spines['bottom'].set_bounds(1, 10)
            ax[c].set_xticks(np.arange(2, 12, 2))
        else:
            ax[c].set_xticks([])
            ax[c].spines[['bottom', 'left', 'right', 'top']].set_visible(False)
        if c in (0, 2, 4):
            ax[c].set_ylabel("Relative amount of content", labelpad=8)
        ax[c].set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)

    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax


def figure_3_combtypes_across_nest(df_mergedinsp: pd.DataFrame,
                                   maxweek: int = 11, out: str | None = None):
    """Figure 3C — worker/drone comb distribution across all nests pooled.

    Same ridgeline engine as :func:`figure_3_contents_across_nest`, but
    all DD colonies are averaged together and the two series are comb
    types (worker grey, drone steelblue), reoriented with
    ``frame_RO`` (forward for DD6/DD7).
    """
    data = df_mergedinsp[(df_mergedinsp['treatment'] == 'DD')
                         & (df_mergedinsp['week'] < maxweek + 1)].copy()
    data = data[['colony', 'week', 'frame', 'totalworkercomb',
                 'totaldronecomb', 'totalcomb']]
    data['frame_RO'] = 11 - data['frame']
    data.loc[(data.colony == 'DD6') | (data.colony == 'DD7'),
             'frame_RO'] = data['frame']
    data = data.drop(columns=['colony']).groupby(['week', 'frame_RO'],
                                                 as_index=False).mean()

    fig, ax = plt.subplots(figsize=(4, 6))
    _ridgeline(data, ax, 'frame_RO',
               [('totalworkercomb', 'grey', np.max(data.totalworkercomb)),
                ('totaldronecomb', 'steelblue', np.max(data.totaldronecomb))],
               yoffset=.9, lw=4)

    legend_elements = [
        Line2D([0], [0], marker='o', color='grey', label='worker',
               markersize=10, lw=0),
        Line2D([0], [0], marker='o', color='steelblue', label='drone',
               markersize=10, lw=0),
    ]
    for w in range(1, int(data.week.max()) + 1):
        label = 'week\n%s' % w if w == 1 else str(w)
        ax.text(np.min(data.frame_RO) - np.max(data.frame_RO) * .08,
                .1 - .9 * w, label, zorder=1000, ha='center')
    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(1, 1.03), frameon=False)
    ax.set_xlabel("Comb position across nest")
    ax.set_ylabel("Relative amount of comb type")
    ax.set_yticks([])
    ax.set_xticks(np.arange(2, 12, 2))
    sns.despine(left=True)

    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax
