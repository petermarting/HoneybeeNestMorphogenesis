"""Seasonality of nest contents through time (paper Figure 1C–F).

Derived from the paper's ``CombContent_Analysis.ipynb``.  Figure 1A–B are photograph/schematic
panels; the comb-segmentation network behind Figure 1B is not released
with the paper (see ``RESULTS.md``).

Panels (legend, Figure 1C–F): (C) amount of each content in the nest
through time, with the three seasons annotated — spring growth, summer
dearth, fall flow; (D) proportion of the nest occupied by the simplified
contents (empty comb, any brood, pollen, nectar & honey); (E) amount of
worker and drone comb; (F) proportion of drone comb occupied by the
simplified contents.  Means ± 95% bootstrapped CIs (seaborn
``pointplot`` defaults) over the six colonies of one treatment; dead-
colony weeks are excluded.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .constants import colony_deadout_week, contenthex

#: Content columns and colours for Figure 1C.
_COUNTS_COLS = ['eggs', 'larvae', 'pupae', 'pollen', 'nectar', 'honey']
#: Columns/labels for Figure 1D (simplified contents).
_PROPS_COLS = ['empty_comb', 'allbrood', 'pollen', 'nectar_honey']
_PROPS_LABELS = ('empty comb', 'any brood', 'pollen', 'nectar & honey')


def weekly_summary(df_mergedinsp: pd.DataFrame,
                   treatment: str = 'DD') -> pd.DataFrame:
    """Colony × true-week sums of comb-count columns for one treatment.

    Excludes dead-colony weeks (``constants.colony_deadout_week``), exactly
    as every panel of the seasonality section does (the notebook recomputed
    this per panel; computing it once yields identical frames).
    """
    data = df_mergedinsp[(df_mergedinsp['treatment'] == treatment) &
                         (~df_mergedinsp.true_week.isin(colony_deadout_week))]
    return data.groupby(by=['colony', 'true_week']).sum(
        numeric_only=True).reset_index()


def month_ticks(df_mergedinsp: pd.DataFrame) -> pd.DataFrame:
    """(true_week, month) tick labels for the Figure 1 time axis.

    First photographed week of each month, April pinned to week 0 — the
    notebook's month-axis logic (which mutated ``df_mergedinsp`` in place;
    here a copy is used).
    """
    df = df_mergedinsp.copy()
    df['month'] = pd.to_datetime(df['date_formatted']).dt.month_name().str[:3]
    df = df.groupby(by=['month']).first().reset_index().sort_values(
        by='true_week')
    df = df[~(df['true_week'].isin(colony_deadout_week))]
    df = df[['true_week', 'month']].reset_index(drop=True)
    df.loc[df['month'] == 'Apr', 'true_week'] = 0
    return df


def figure_1_seasonality(df_mergedinsp: pd.DataFrame, treatment: str = 'DD',
                         out: str | None = None):
    """Four-panel seasonality composite (Figure 1C–F).

    Parameters
    ----------
    df_mergedinsp
        Comb-count table with derived columns (``data.load_tables()``).
    treatment
        ``'DD'`` (unshuffled control, the paper figure) or ``'SH'``.
    out
        Optional path; when given the figure is saved there.

    Returns
    -------
    (fig, axes)
    """
    # NOTE(season): seaborn pointplot CIs bootstrap-approximate the mean at
    # every week; they jitter between unseeded runs (paper shows mean ± CI).
    data = weekly_summary(df_mergedinsp, treatment=treatment)

    fig, ax = plt.subplots(4, 1, figsize=(10, 15.5), sharex=True)

    # ### CONTENT COUNTS (Fig 1C) ###
    for i, content in enumerate(_COUNTS_COLS):
        color = 'lightgrey' if i == len(_COUNTS_COLS) - 1 \
            else contenthex[i + 3]
        sns.pointplot(ax=ax[0], data=data, x='true_week',
                      y=data[content] / 1000, native_scale=True,
                      label=content, color=color)
    sns.despine()
    ax[0].set_ylabel('Number of cells (thousands)')
    ax[0].legend(loc='center', bbox_to_anchor=(.52, .7), frameon=False)

    # Season labels
    labely = 17
    for name, span, widthb in (('spring growth', [1, 10], 8.8),
                               ('summer dearth', [10, 23], 10),
                               ('fall flow', [23, 30], 7.1)):
        ax[0].annotate(name, xy=(np.mean(span), labely),
                       xytext=(np.mean(span), labely + labely * .03),
                       ha='center', va='bottom',
                       arrowprops=dict(arrowstyle='-[, widthB=%s, lengthB=1'
                                       % widthb, lw=2.0, color='k'))

    # ### CONTENT PROPORTIONS (Fig 1D) ###
    props = weekly_summary(df_mergedinsp, treatment=treatment)
    props['nectar_honey'] = props['nectar'] + props['honey']
    for i, content in enumerate(_PROPS_COLS):
        props['prop'] = props[content] / props['totalcomb']
        color = contenthex[2] if i == 0 else contenthex[i + 4]
        sns.pointplot(ax=ax[1], data=props, x='true_week', y='prop',
                      native_scale=True, label=_PROPS_LABELS[i], color=color)
    ax[1].set_ylabel('Proportion of nest (simplified contents)')
    ax[1].legend(loc='center', bbox_to_anchor=(.55, .5), frameon=False)

    # ### DRONE COMB (Fig 1E) ###
    combs = weekly_summary(df_mergedinsp, treatment=treatment)
    for i, (content, color, label) in enumerate(zip(
            ['totalworkercomb', 'totaldronecomb'], ['k', 'grey'],
            ['worker comb', 'drone comb'])):
        sns.pointplot(ax=ax[2], data=combs, x='true_week',
                      y=combs[content] / 1000, native_scale=True,
                      label=label, color=color)
    ax[2].set_ylabel('Number of cells (thousands)')
    ax[2].legend(loc='center', bbox_to_anchor=(.55, .5), frameon=False)

    # ### DRONE CONTENT (Fig 1F) ###
    drones = weekly_summary(df_mergedinsp, treatment=treatment)
    drones['drone_nectar_honey'] = drones['drone_nectar'] + drones['drone_honey']
    for i, (content, label) in enumerate(zip(
            ['drone_allbrood', 'drone_pollen', 'drone_nectar_honey'],
            ['any brood', 'pollen', 'nectar & honey'])):
        drones['prop'] = drones[content] / drones['totaldronecomb']
        sns.pointplot(ax=ax[3], data=drones, x='true_week', y='prop',
                      native_scale=True, label=label,
                      color=contenthex[i + 5])
    ax[3].set_ylabel('Proportion of drone comb')
    ax[3].set_xlabel('Time (weeks)')
    ax[3].legend(loc='center', bbox_to_anchor=(.55, .7), frameon=False)

    # Shaded region highlighting the spring-growth window across panels
    rect = mpatches.Rectangle([.397, .11], 0.282, 0.766, ec=None, fc='k',
                              alpha=0.06, clip_on=False, zorder=0,
                              transform=fig.transFigure)
    ax[3].add_patch(rect)

    # Month labels along the bottom axis
    ticks = month_ticks(df_mergedinsp)
    ax_month = ax[3].twiny()
    ax_month.set_xlim(ax[3].get_xlim())
    ax_month.set_xticks(ticks['true_week'])
    ax_month.set_xticklabels(ticks['month'])
    ax_month.xaxis.set_ticks_position('bottom')
    ax_month.xaxis.set_label_position('bottom')
    ax_month.spines['bottom'].set_position(('outward', 40))
    ax_month.set_xlabel('')
    sns.despine()

    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax
