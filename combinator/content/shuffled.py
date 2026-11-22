"""Shuffled-vs-control nest comparisons (paper Figure 5B–G).

Derived from the paper's ``CombContent_Analysis.ipynb``.

Panels (legend, Figure 5): (A) schematic of the shuffling regime —
drawn, no code; (B) amount of bees at each life stage per colony-week
(bootstrapped median ± 95% CI; shuffled significantly more eggs);
(C) egg/larval developmental fate tracked week-to-week from the comb
masks (failed progressions in red); (D) week-to-week fluctuation of
per-comb contents; (E) content ridgelines for an example control (DD2)
and shuffled (SH2) colony by physical comb ID; (F) comb-specialization
counts, control vs shuffled (binomial tests); (G) comb-specialization
locations for all 12 colonies.

All bootstraps are unseeded hierarchical bootstraps (n = 4000); paper
values are approximate and CI medians jitter between runs.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import binomtest

from . import util
from .constants import (colony_deadout_week, colony_names, content_names,
                        contenthex, treatmenthex)

_LIFE_STAGES = ['eggs', 'larvae', 'pupae', 'bees_onframe']


# ---------------------------------------------------------------------------
# Life stages (Figure 5B)
# ---------------------------------------------------------------------------

def life_stage_bootstraps(df_mergedinsp: pd.DataFrame, maxweek: int = 11,
                          n_bootstraps: int = 4000):
    """Simple (flat) bootstrap of colony-week life-stage totals for DD/SH.

    Returns ``(medians, errors)`` indexed ``[treatment][content]``
    (treatment order DD, SH) for the first ``maxweek - 1`` weeks.
    """
    hierarchy_levels: list[str] = []
    medians, errors = [], []
    for treat in ['DD', 'SH']:
        t_medians, t_errors = [], []
        for content in _LIFE_STAGES:
            target = content
            data = df_mergedinsp[(df_mergedinsp['treatment'] == treat)
                                 & (df_mergedinsp['week'] < maxweek)]
            data = data[np.unique(['colony', 'week'] + [target])]
            data = data.groupby(['colony', 'week']).sum(
                numeric_only=True).reset_index()
            bootstrap = util.hierarchical_bootstrap(
                data, hierarchy_levels, method='mean',
                n_bootstraps=n_bootstraps)
            summary = util.bootstrap_summary_stats(bootstrap[target].values)
            t_medians.append(summary['median'])
            t_errors.append(summary['errors'])
        medians.append(t_medians)
        errors.append(t_errors)
    return medians, errors


def figure_5_life_stages(df_mergedinsp: pd.DataFrame, medians, errors,
                         out: str | None = None):
    """Figure 5B — life-stage amounts per colony-week, control vs shuffled."""
    treatlabels = ['control', 'shuffled']
    data = df_mergedinsp[(~df_mergedinsp.true_week.isin(colony_deadout_week))
                         & (df_mergedinsp['week'] < 11)]
    data = data.groupby(by=['colony', 'treatment', 'week',
                            'true_week']).sum(numeric_only=True).reset_index()
    data = data[['colony', 'treatment', 'week', 'true_week'] + _LIFE_STAGES]
    data = pd.melt(data, id_vars=['colony', 'treatment', 'week', 'true_week'],
                   value_vars=_LIFE_STAGES, var_name='content',
                   value_name='count')
    data['count'] = data['count'] / 1000

    fig, ax0 = plt.subplots(figsize=(4, 4))
    sns.stripplot(ax=ax0, data=data, x='content', y='count',
                  hue='treatment', palette=treatmenthex, alpha=.5,
                  dodge=True, zorder=1)

    for i in range(2):
        offset = i * .4 - .2
        ax0.errorbar(x=np.arange(len(medians[i])) + offset,
                     y=np.array(medians[i]) / 1000,
                     yerr=(np.array(errors[i]) / 1000).T, fmt='o',
                     color='k', capsize=3, zorder=10000)

    sns.despine()
    ax0.set_xlabel('Life stage')
    ax0.set_ylabel('Amount in nest per week (thousands)')
    ax0.set_xticks(ticks=[0, 1, 2, 3], labels=['eggs', 'larvae', 'pupae',
                                               'adults'])
    a = Line2D([], [], color=treatmenthex[0], label=treatlabels[0],
               marker='o', linestyle='None')
    b = Line2D([], [], color=treatmenthex[1], label=treatlabels[1],
               marker='o', linestyle='None')
    ax0.legend(handles=[a, b], frameon=False)

    y_offset = .1
    annotations = ['*', 'n.s.', 'n.s.', 'n.s.']
    for i, category in enumerate(_LIFE_STAGES):
        subset = data[data['content'] == category]
        y_max = subset['count'].max()
        ax0.text(category, y_max + y_offset, annotations[i], ha='center',
                 va='bottom', color='black', fontsize=16)

    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax0


# ---------------------------------------------------------------------------
# Brood survival (Figure 5C)
# ---------------------------------------------------------------------------

def brood_flow(Contents: np.ndarray, maxweek: int = 10) -> pd.DataFrame:
    """Week-to-week cell-fate transitions for brood-relevant contents.

    For consecutive weeks, tracks every cell that was empty/egg/larva
    (side-alternating sample of each comb face) and records its content
    one week later; empty cells that merely stayed empty or turned into
    non-brood contents are dropped.  ~20 s.
    """
    appended = []
    side = 'a'
    for c in range(12):
        for w in range(maxweek - 1):
            for f in range(20):
                mask0 = Contents[c][w][f]
                mask1 = Contents[c][w + 1][f]

                mask0 = np.where(mask0 < 2, 0, mask0)  # keep empty/eggs/larvae
                mask0 = np.where(mask0 > 4, 0, mask0)
                mask1 = np.where(mask0 > 0, mask1, 0)
                mask0 = np.where((mask0 == 2) & (mask1 < 3), 0, mask0)
                mask0 = np.where((mask0 == 2) & (mask1 > 5), 0, mask0)
                coords = np.argwhere(mask0 > 0)

                sources, targets = [], []
                for i in coords:
                    sources.append(mask0[i[0]][i[1]])
                    targets.append(mask1[i[0]][i[1]])

                temp = pd.DataFrame()
                temp['colony'] = [colony_names[c]] * len(sources)
                temp['treatment'] = [colony_names[c][:2]] * len(sources)
                temp['week'] = [w + 1] * len(sources)
                temp['frame'] = [int(f / 2) + 1] * len(sources)
                temp['side'] = [side] * len(sources)
                temp['source'] = sources
                temp['target'] = targets
                temp['source_name'] = np.take(content_names, sources)
                temp['target_name'] = np.take(content_names, targets)
                appended.append(temp)

                side = 'b' if side == 'a' else 'a'
    return pd.concat(appended).reset_index(drop=True)


def brood_outcomes(df_broodflow: pd.DataFrame) -> pd.DataFrame:
    """Classify each tracked transition as success or failure (cell 67).

    Failure = brood became a storage/empty/background content, an egg
    that did not progress, or a reverted larva.
    """
    df = df_broodflow.copy()
    df['target_outcome'] = df['target_name']
    df.loc[df['target'] > 5, 'target_outcome'] = 'failed'
    df.loc[df['target'] < 3, 'target_outcome'] = 'failed'
    df.loc[(df['source'] == 3) & (df['target'] == 3), 'target_outcome'] = 'failed'
    df.loc[(df['source'] == 4) & (df['target'] == 3), 'target_outcome'] = 'failed'
    df['outcome'] = df['target']
    df.loc[df['target_outcome'] == 'failed', 'outcome'] = 0
    return df


def brood_flow_summary(df_broodflow: pd.DataFrame) -> pd.DataFrame:
    """Transition counts per colony/treatment/source/outcome (``transition`` column)."""
    df_sum = df_broodflow.groupby(
        ['colony', 'treatment', 'source_name',
         'target_outcome']).size().reset_index(name='value')
    df_sum['transition'] = df_sum['source_name'] + '_' + df_sum['target_outcome']
    return df_sum


def figure_5_brood_survival(df_sum: pd.DataFrame, out: str | None = None):
    """Figure 5C — total eggs/larvae over 10 weeks with failed progressions."""
    fig, ax = plt.subplots(1, 2, figsize=(4, 4), constrained_layout=True)

    def panel(axp, source):
        data = df_sum[df_sum['transition'].isin(
            [f'{source}_failed', f'{source}_larvae', f'{source}_pupae'])].copy()
        data.loc[data['transition'] != f'{source}_failed',
                 'target_outcome'] = 'succeeded'
        df_totals = data.groupby(['treatment']).sum(
            numeric_only=True).reset_index()
        df_outcomes = data.groupby(['treatment', 'target_outcome']).sum(
            numeric_only=True).reset_index()
        sns.barplot(ax=axp, data=df_totals, x='treatment',
                    y=df_totals['value'] / 1000, color='red')
        df = df_outcomes[df_outcomes['target_outcome'] == 'succeeded']
        sns.barplot(ax=axp, data=df, x='treatment', y=df['value'] / 1000,
                    palette=treatmenthex)
        sns.despine(ax=axp)
        axp.set_xlabel('Treatment')
        axp.set_title(source.capitalize())
        axp.set_xticklabels(['cont', 'shuf'])

    panel(ax[0], 'eggs')
    ax[0].set_ylabel('Number of cells (thousands)')
    panel(ax[1], 'larvae')
    ax[1].set_ylabel('')

    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax


# ---------------------------------------------------------------------------
# Content fluctuations (Figure 5D)
# ---------------------------------------------------------------------------

def content_change_frames(df_mergedinsp: pd.DataFrame) -> pd.DataFrame:
    """Add week-to-week per-comb content-change columns (cell 71).

    ``ttl_abs_change`` = sum of |Δ| of the six content counts relative to
    the same colony/frame/side in the previous week; ``perc_ttl_abs_change``
    normalizes by comb size.  Row order (colony, frame, side, week) is
    enforced internally; the input frame is not mutated.
    """
    df = df_mergedinsp.sort_values(
        by=['colony', 'frame', 'side', 'week']).reset_index(drop=True).copy()
    df['ttl_abs_change'] = 0
    for content in content_names[3:9]:
        df[content + '_change'] = df[content] - df[content].shift(+1)
        df.loc[df.week == 1, content + '_change'] = np.nan
        df['perc_' + content + '_change'] = (
            df['perc_' + content] - df['perc_' + content].shift(+1))
        df.loc[df.week == 1, 'perc_' + content + '_change'] = np.nan
        df['ttl_abs_change'] = df['ttl_abs_change'] + abs(
            df[content + '_change'])
    df['perc_ttl_abs_change'] = (df['ttl_abs_change'] / df['totalcomb']) * 100
    return df


def content_change_bootstrap(df_changes: pd.DataFrame,
                             var: str = 'perc_ttl_abs_change',
                             maxweek: int = 11, min_worker_comb: float = 200.,
                             n_bootstraps: int = 4000):
    """Colony-level hierarchical bootstrap of a fluctuation variable, DD vs SH."""
    hierarchy_levels = ['colony']
    medians, errors = [], []
    for treat in ['DD', 'SH']:
        data = df_changes[(df_changes['treatment'] == treat)
                          & (df_changes['week'] < maxweek)
                          & (df_changes['totalworkercomb'] > min_worker_comb)]
        data = data[hierarchy_levels + [var]]
        bootstrap = util.hierarchical_bootstrap(
            data, hierarchy_levels, method='mean',
            n_bootstraps=n_bootstraps)
        summary = util.bootstrap_summary_stats(bootstrap[var].values)
        medians.append(summary['median'])
        errors.append(summary['errors'])
    return medians, errors


def figure_5_content_fluctuations(df_changes: pd.DataFrame, errors,
                                  out: str | None = None):
    """Figure 5D — weekly change in per-comb contents, control vs shuffled.

    ``errors`` comes from :func:`content_change_bootstrap` with the
    default ``perc_ttl_abs_change`` variable.
    """
    data = df_changes[(df_changes['week'] > 1)
                      & (df_changes['week'] < 11)
                      & (df_changes['totalworkercomb'] > 200)].copy()
    data['treatment'] = data['treatment'].replace({'DD': 'control',
                                                   'SH': 'shuffled'})

    fig, ax = plt.subplots(1, 2, figsize=(5, 4), width_ratios=[2.5, 1],
                           sharey=False)
    sns.pointplot(ax=ax[0], data=data, y=data['perc_ttl_abs_change'],
                  x='week', hue='treatment', palette='viridis', dodge=True,
                  native_scale=True)
    ax[0].legend(frameon=False)
    ax[0].set_ylabel('Weekly change in cell contents on comb (%)')

    sns.stripplot(ax=ax[1], data=data, y=data['perc_ttl_abs_change'],
                  x='treatment', hue='treatment', palette='viridis',
                  legend=False, jitter=0.2, alpha=.2)
    means = data.groupby('treatment').agg(
        {'perc_ttl_abs_change': 'mean'}).reset_index()
    ax[1].errorbar(x=np.arange(len(means)), y=means['perc_ttl_abs_change'],
                   yerr=np.array(errors).squeeze(), fmt='o', color='k',
                   capsize=3, zorder=1000000)
    ax[1].set_ylabel('')
    ax[1].set_xticklabels(['cont', 'shuf'])
    ax[1].set_xlim(-0.5, 1.5)
    ax[1].text(0.5, max(data['perc_ttl_abs_change']) * .9, '*', ha='center',
               va='bottom', fontsize=15)
    sns.despine()

    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax


# ---------------------------------------------------------------------------
# Example ridgelines (Figure 5E)
# ---------------------------------------------------------------------------

def figure_5_example_ridgelines(df_mergedinsp: pd.DataFrame,
                                colonies: tuple = (1, 6), maxweek: int = 11,
                                out: str | None = None):
    """Figure 5E — content ridgelines for example control (DD2) and shuffled (SH2) colonies.

    x axis = physical comb ID (combs in shuffled nests move, so identity,
    not position, is the meaningful axis here).
    """
    yoffset = .8
    lw = 2.5
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 4.5))

    for c, col_idx in enumerate(colonies):
        data = df_mergedinsp[(df_mergedinsp['colony'] == colony_names[col_idx])
                             & (df_mergedinsp['week'] < maxweek + 1)]
        data = data[['colony', 'week', 'frame', 'pollen', 'nectar_honey',
                     'allbrood', 'totalcomb']].reset_index(
            drop=True).groupby(['colony', 'week', 'frame'],
                               as_index=False).mean()

        maxpupae = np.max(data.allbrood)
        maxpollen = np.max(data.pollen)
        maxnectar = np.max(data.nectar_honey)

        cmap = mpl.colormaps.get_cmap('viridis')
        Nlen = int(np.max(data.week) + 2 - (np.min(data.week) - 2))
        colors = cmap(np.linspace(0, 1, Nlen))

        for w in range(int(np.max(data.week))):
            dfw = data[data['week'] == w]
            if len(dfw) == 0:
                continue
            # active nest edges
            comb_last = 0
            for i in range(len(dfw)):
                comb = dfw.totalcomb.iloc[i]
                if (comb > comb_last) & (comb_last < 100):
                    nest_edge_L = dfw.frame.iloc[i]
                    break
            comb_last = 0
            for i in range(len(dfw) - 1, -1, -1):
                comb = dfw.totalcomb.iloc[i]
                if (comb > 100) & (comb_last < 100):
                    nest_edge_R = dfw.frame.iloc[i]
                    break
            dfw = dfw[(dfw.frame >= min(nest_edge_L, nest_edge_R))
                      & (dfw.frame <= max(nest_edge_L, nest_edge_R))]

            x = 11 - np.array(dfw.frame)
            for j, (col, colmax, color) in enumerate(
                    [('allbrood', maxpupae, colors[7]),
                     ('nectar_honey', maxnectar, colors[4]),
                     ('pollen', maxpollen, colors[10])]):
                y = np.array(dfw[col]) / colmax - yoffset * w
                zorder = w * 5 + j
                ax[c].fill_between(x, y, 0 - yoffset * w, alpha=1, color=color,
                                   zorder=zorder, interpolate=False)
                ax[c].plot(x, y, c="w", lw=lw, zorder=zorder - 1)

            if c == 0:
                legend_elements = [
                    Line2D([0], [0], marker='o', color=colors[10],
                           label='pollen', markersize=10, lw=0),
                    Line2D([0], [0], marker='o', color=colors[7],
                           label='any brood', markersize=10, lw=0),
                    Line2D([0], [0], marker='o', color=colors[4],
                           label='nectar\n& honey', markersize=10, lw=0),
                ]
                ax[c].legend(handles=legend_elements, loc='upper right',
                             bbox_to_anchor=(1.05, 1), edgecolor='w',
                             facecolor='w', framealpha=.7,
                             handletextpad=-.2).set_zorder(1000)
            label = 'week\n%s' % w if w == 1 else str(w)
            ax[c].text(np.min(x) - np.max(x) * .08, 0 - yoffset * w, label,
                       zorder=1000, ha='center')
        ax[c].set_xlim(np.min(x) - np.max(x) * .1)
        ax[c].set_xlabel("Comb ID")
        ax[c].spines[['left', 'right', 'top']].set_visible(False)
        ax[c].spines['bottom'].set_bounds(1, 10)
        ax[c].set_xticks(np.arange(2, 12, 2))
        if c == 0:
            ax[c].set_ylabel("Relative amount of content", labelpad=8)
        ax[c].set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)

    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax


# ---------------------------------------------------------------------------
# Comb specialization, all colonies (Figures 5F, 5G)
# ---------------------------------------------------------------------------

def comb_specialization_all(Contents: np.ndarray, maxweek: int = 10):
    """Comb specialization for all 12 colonies using DD-derived thresholds.

    Cell thresholds from the DD colonies, comb thresholds from the DD
    combs, both then applied to all colonies (~33 s).
    """
    cell_sp = util.cell_specialists(Contents[:6], maxweek=10,
                                    prop_thres=None)
    comb_sp_dd = util.comb_specialists(Contents[:6], maxweek=maxweek,
                                       cell_prop_thres=cell_sp.prop_thres,
                                       comb_prop_thres=None,
                                       print_results=True)
    return util.comb_specialists(Contents, maxweek=maxweek,
                                 cell_prop_thres=cell_sp.prop_thres,
                                 comb_prop_thres=comb_sp_dd.comb_prop_thres,
                                 print_results=True)


def figure_5_comb_specialization(comb_sp, out: str | None = None):
    """Figures 5F/G — specialization counts (binomial tests) and locations.

    NOTE: the paper text's "55.9% fewer specialized combs" matches the
    released pipeline exactly (control 34 vs shuffled 15 specialists)
    once the unbuilt DD6 frame-10 comb is excluded from the control
    count — the notebook itself classifies that phantom comb (as nectar)
    and only blanks it visually; including it gives 35 and 57.1%.
    """
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.8), width_ratios=[1.3, 1],
                           constrained_layout=True)

    categories = [0, 2, 5, 6, 7]
    DDcounts = [np.count_nonzero(comb_sp.comb_ids[:6] == id)
                for id in categories]
    SHcounts = [np.count_nonzero(comb_sp.comb_ids[6:] == id)
                for id in categories]

    classification_names = ['general', 'empty', 'brood', 'pollen', 'nectar']
    x = np.arange(len(classification_names))
    y1 = np.array(DDcounts) + .2
    y2 = np.array(SHcounts) + .2
    width = 0.40
    ax[0].bar(x - 0.2, y1, width, color=treatmenthex[0], alpha=1)
    ax[0].bar(x + 0.2, y2, width, color=treatmenthex[1], alpha=1)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(classification_names)
    ax[0].set_xlabel('Comb specialization')
    ax[0].set_ylabel('Comb count')
    ax[0].legend(["Control", "Shuffled"], frameon=False)

    for i in range(len(classification_names)):
        dd_count = DDcounts[i]
        sh_count = SHcounts[i]
        result = binomtest(dd_count, n=dd_count + sh_count, p=0.5)
        label = '*' if result.pvalue < 0.05 else 'n.s.'
        ax[0].text(x[i], max(dd_count, sh_count) + .5, label, ha='center',
                   va='bottom', fontsize=14 if label == '*' else 12)

    # specialist locations for all colonies
    for c in range(len(comb_sp.comb_ids)):
        for f in range(len(comb_sp.comb_ids[c])):
            id = comb_sp.comb_ids[c][f]
            x = f + 1
            y = 13 - (c + 1) if c > 5 else 13 - c
            if c == 4 and f == 9:
                color = 'w'
            else:
                color = contenthex[id]
            ax[1].scatter(x=x, y=y, c=color, s=210, marker='s')

    ax[1].set_xlabel('Comb ID')
    ax[1].set_ylabel('Shuffled colony IDs        Control colony IDs')
    ax[1].set_yticks(np.arange(1, 14))
    ax[1].set_yticklabels(np.concatenate(
        (colony_names[:6], [''], colony_names[6:]))[::-1])
    ax[1].legend(['nectar', 'pollen', 'brood', 'empty', 'generalist'],
                 title='specialization', bbox_to_anchor=(1.5, .72),
                 loc='upper right', frameon=False)
    leg = ax[1].get_legend()
    colors = [7, 6, 5, 2, 0]
    for i in range(len(colors)):
        leg.legend_handles[i].set_facecolor(contenthex[colors[i]])
        leg.legend_handles[i].set_edgecolor('w')
    sns.despine()

    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax
