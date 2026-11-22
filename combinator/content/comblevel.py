"""Comb-level organization (paper Figure 3A, 3B).

Derived from the paper's ``CombContent_Analysis.ipynb``.

Panels (legend, Figure 3A/B): (A) PCA of 10 comb-level features over
time — comb trajectories through the PC1–PC2 plane with example comb
masks and the PC loadings (PC1 ~40% variance: diversity/complexity/
abundance; PC2 ~24%: clustering vs pollen); (B) vertical distribution of
simplified contents for combs of increasing size classes (brood below,
pollen middle, nectar/honey top).

NOTE: the normalized two-sample Kolmogorov-Smirnov p-values quoted in
the paper text for these distributions ("brood-pollen p < 0.01 ...
Figure 3B") have **no released code** — the notebook plots the
distributions but never runs the tests (see ``RESULTS.md``).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from sklearn import decomposition

from . import util
from .constants import (colony_deadout_week, colony_names, content_names,
                        simplehex)

#: The 10 comb-level PCA features (notebook cell 35: 4 makeup, 3
#: size/shape, 3 landscape).
PCA_VARIABLES = [
    'totalcomb',                   # size/shape
    'perc_empty_comb',             # makeup
    'perc_pollen',                 # makeup
    'perc_nectar_honey',           # makeup
    'perc_allbrood',               # makeup
    'comb_aspecthw',               # size/shape
    'comb_perim_area_ratio',       # size/shape
    'landscape_shape_index',       # landscape
    'shannon_diversity_index',     # landscape
    'dispersion_metric',           # landscape
]


def comb_pca(df_mergedinsp: pd.DataFrame, treatment: str = 'DD',
             n_components: int = 3):
    """Comb-level PCA of ``PCA_VARIABLES`` (Figure 3A computation).

    Rows = combs (colony × week × frame × side) of one treatment with
    dead-colony weeks and infinite perimeter/area ratios (zero-comb
    frames) excluded; features are log1p-transformed then z-scored.
    Returns ``(pca, loadings, df_pca, explained_variance_ratio)`` where
    ``df_pca`` carries the leading identifier columns + PC scores and has
    the one outlier frame the notebook discards removed (DD1 week 16
    frame 9 side b: way off alone, high pollen and dispersion).
    """
    data = df_mergedinsp[(df_mergedinsp['treatment'] == treatment)
                         & (~df_mergedinsp.true_week.isin(colony_deadout_week))
                         & (df_mergedinsp['comb_perim_area_ratio'] != np.inf)]
    data = data[PCA_VARIABLES]

    # log transform (natural log + 1), then z-score
    df_logtrans = np.log1p(data)
    df_normalized = ((df_logtrans - df_logtrans.mean()) / df_logtrans.std())

    df_head = df_mergedinsp[df_mergedinsp.columns.tolist()[:6]]
    df_combined = pd.concat([df_head, df_normalized], axis=1)

    df_normalized = df_normalized.dropna().reset_index(drop=True)
    df_combined = df_combined.dropna().reset_index(drop=True)

    pca = decomposition.PCA(n_components=n_components)
    pca_values = pca.fit_transform(df_normalized)

    loadings = pd.DataFrame(pca.components_.T,
                            columns=['PC%s' % _ for _ in range(n_components)],
                            index=data.columns)

    df_pca_values = pd.DataFrame({'PC0': pca_values[:, 0],
                                  'PC1': pca_values[:, 1],
                                  'PC2': pca_values[:, 2]})
    df_pca = pd.concat([df_combined[df_combined.columns.tolist()[:6]],
                        df_pca_values], axis=1)

    # NOTE(original): the notebook's next line computed 'frame_reoriented'
    # with `(colony!='DD6')|(colony!='DD7')|(colony!='SH4')` (always True)
    # and never used it; df_mergedinsp['frame_RO'] is the correct column.
    # The dead line is omitted here; no figure or statistic depends on it.

    # remove the one weird frame that is way off on its own
    df_pca = df_pca[~((df_pca['colony'] == 'DD1')
                      & (df_pca['true_week'] == 16)
                      & (df_pca['frame'] == 9)
                      & (df_pca['side'] == 'b'))].reset_index(drop=True)

    return pca, loadings, df_pca, pca.explained_variance_ratio_


def figure_3_comb_pca(df_pca: pd.DataFrame, loadings: pd.DataFrame,
                      explained_variance_ratio: np.ndarray,
                      Contents: np.ndarray, out: str | None = None):
    """Figure 3A — PCA trajectories, example comb masks, and loadings.

    Trajectories connect the mean comb state of consecutive weeks
    (black mean line over all combs, colour = time in weeks); five
    hand-picked comb masks are shown beside the plane with their
    numbered positions.  Loadings bars display the sign-flattened
    coefficients (|loading| < 0.2 zeroed, as in the notebook).
    """
    data = df_pca

    # flatten small loadings for display only (notebook cells 38/39)
    loadings = loadings.copy()
    loadings.loc[(loadings.PC0 > -.2) & (loadings.PC0 < .2), 'PC0'] = 0
    loadings.loc[(loadings.PC1 > -.2) & (loadings.PC1 < .2), 'PC1'] = 0

    colormap = 'cividis_r'
    Nmax = np.max(data.true_week) + 1
    Nmin = np.min(data.true_week) - 1
    Nlen = int(Nmax - Nmin)
    cmap = mpl.colormaps.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, Nlen))

    fig = plt.figure(figsize=(10, 7), constrained_layout=False)

    # PCA points -------------------------------------------------------
    gs1 = GridSpec(2, 2, left=.35, right=1, top=1, bottom=.1, figure=fig,
                   wspace=.05, hspace=.2, width_ratios=[1, .1],
                   height_ratios=[1, .1])
    ax1 = fig.add_subplot(gs1[0, 0])
    for c in range(6):
        for f in range(20):
            for side in ['a', 'b']:
                df = data[(data['colony'] == colony_names[c])
                          & (data['frame'] == f + 1)
                          & (data['side'] == side)]
                for w in df.week.unique().tolist()[1:]:
                    df2 = df[(df['week'] == w - 1) | (df['week'] == w)]
                    ax1.plot(df2.PC0, df2.PC1, marker='o',
                             color=colors[int(df2.true_week.unique().tolist()[0])],
                             alpha=.4)

    df = data[['week', 'true_week', 'PC0', 'PC1']].groupby(
        by=['week', 'true_week']).mean().reset_index()
    for i in df.week.unique().tolist()[1:]:
        df2 = df[(df.week == i - 1) | (df.week == i)]
        ax1.plot(df2.PC0, df2.PC1, marker='o', ms=8, lw=6, color='k',
                 zorder=10000)
        ax1.plot(df2.PC0, df2.PC1, marker='o', ms=6, lw=3.5,
                 color=colors[int(df2.true_week.unique().tolist()[0])],
                 zorder=10001)
    ax1.set_xlabel('→ More diversity, complexity, abundance (PC1 '
                   + str(round(explained_variance_ratio[0] * 100)) + '%) →')
    ax1.set_ylabel('→ More clustering, less pollen (PC2 '
                   + str(round(explained_variance_ratio[1] * 100)) + '%) →')
    sns.despine()

    # colourbar (time)
    norm = mpl.colors.Normalize(vmin=Nmin, vmax=Nmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ticks = np.arange(round(Nmin, -2), Nmax, round(Nlen / 4, -1))
    cbar_ax = fig.add_axes([.54, .34, .13, .02])
    cbar = plt.colorbar(sm, ticks=ticks, shrink=.4, orientation='horizontal',
                        cax=cbar_ax)
    cbar.set_label("Time (weeks)")

    # Example frames ----------------------------------------------------
    gs2 = GridSpec(5, 1, left=0, right=.33, top=1, bottom=0, figure=fig,
                   hspace=.05)
    sides = ['a', 'b']
    cs = [4, 1, 0, 0, 0]
    ws = [1, 8, 6, 12, 16]
    fs = [1, 6, 9, 2, 5]
    ss = [0, 1, 0, 1, 1]
    for i in range(5):
        ax = fig.add_subplot(gs2[i])
        df = data[(data['colony'] == colony_names[cs[i]])
                  & (data['week'] == ws[i] + 1)
                  & (data['frame'] == fs[i] + 1)
                  & (data['side'] == sides[ss[i]])]
        mask = Contents[cs[i]][ws[i]][fs[i] * 2 + ss[i]]
        mask = np.where(mask > 8, 2, mask)  # cap at 8: queen/bee -> empty
        ax1.text(float(df.PC0.iloc[0]), float(df.PC1.iloc[0]), s=str(i + 1),
                 c='k', ha='center', va='center', weight='bold', fontsize=14,
                 path_effects=[pe.withStroke(linewidth=2, foreground="w")],
                 zorder=10002)
        ax.imshow(mask, cmap='gist_stern', vmin=0, vmax=8)
        ax.axis('off')
        ax.text(mask.shape[1] * .05, mask.shape[0] * .12, str(i + 1), c='k',
                ha='center', va='center', weight='bold', fontsize=14,
                path_effects=[pe.withStroke(linewidth=2, foreground="w")])

    # PC loadings ---------------------------------------------------------
    ax = fig.add_subplot(gs1[1, 0])
    sns.barplot(ax=ax, data=loadings, y='PC0', x=loadings.index, hue='PC0',
                palette='coolwarm', orient='v',
                order=loadings.sort_values(by=['PC0'],
                                           ascending=True).index.tolist())
    ax.legend([], [], frameon=False)
    ax.set_ylabel('loading\nscore', fontsize=14)
    ax.set_xlabel('')
    ax.axhline(-.2, color='k', linestyle='--', alpha=.2)
    ax.axhline(0, color='k')
    ax.axhline(.2, color='k', linestyle='--', alpha=.2)
    labels = ['% empty', 'comb size', '% pollen', 'comb aspect', 'comb perim.',
              'dispersion', '% nectar', 'landscape s.i.', '% brood',
              "shannon d.i."]
    ax.set_yticks([])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = fig.add_subplot(gs1[0, 1])
    sns.barplot(ax=ax, data=loadings, x='PC1', y=loadings.index, hue='PC1',
                palette='coolwarm', orient='h',
                order=loadings.sort_values(by=['PC1'],
                                           ascending=False).index.tolist())
    ax.legend([], [], frameon=False)
    ax.set_xlabel('loading\nscore', fontsize=14)
    ax.set_ylabel('')
    ax.axvline(-.2, color='k', linestyle='--', alpha=.2)
    ax.axvline(0, color='k')
    ax.axvline(.2, color='k', linestyle='--', alpha=.2)
    labels = ['clustering', '% nectar', 'comb size', '% empty', '% brood',
              'comb aspect', 'comb perim.', "shannon d.i.", 'landscape s.i.',
              '% pollen']
    ax.set_xticks([])
    ax.yaxis.tick_right()
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # loading colourbar
    colormap = 'coolwarm'
    Nmax = np.max(loadings.PC0)
    Nmin = np.min(loadings.PC0)
    cmap = mpl.colormaps.get_cmap(colormap)
    norm = mpl.colors.Normalize(vmin=Nmin, vmax=Nmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ticks = [-.4, 0, .4]
    cbar_ax = fig.add_axes([.98, .125, .08, .02])
    cbar = plt.colorbar(sm, ticks=ticks, shrink=.4, orientation='horizontal',
                        cax=cbar_ax)
    cbar.set_label("Loading score")
    cbar.ax.set_xticklabels(['-0.4', '0', '0.4'])

    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0)
    return fig


def vertical_distribution_frames(Contents: np.ndarray, maxweek: int = 10):
    """Cell-coordinate frames for the vertical/horizontal distribution (Fig 3B).

    Extracts the 3D coordinates of every comb cell (all 12 colonies,
    first ``maxweek`` weeks; contents empty_comb..honey) with per-comb
    size/height/width and position normalised to the comb extent, plus
    the notebook's reorientations: ``z_RO`` orders frames from the nest
    initiation wall (forward for DD6/DD7/SH4), ``x_RO``/``x_RO_prop``
    mirror each comb face so low x is closer to the side where the comb
    was initiated (detected from which vertical edge the comb spans to).
    Takes ~1 minute.
    """
    appended = []
    for c in range(12):
        for w in range(maxweek):
            df_temp = util.xyz_coords(
                Contents[c][w], content_names[2:9], contents=np.arange(2, 9),
                colony=colony_names[c], week=w)
            appended.append(df_temp)

    df_coords = pd.concat(appended).reset_index(drop=True)
    df_coords['week'] = df_coords['week'] + 1
    df_coords['treatment'] = df_coords['colony'].astype(str).str[:2]

    # comb size
    df_coords['totalcomb'] = df_coords.groupby(
        ['colony', 'week', 'z'])['content'].transform('count')

    # simplified contents
    df_coords['content_simple'] = df_coords['content']
    df_coords.loc[(df_coords.content == 'eggs')
                  | (df_coords.content == 'larvae')
                  | (df_coords.content == 'pupae'), 'content_simple'] = 'any_brood'
    df_coords.loc[(df_coords.content == 'honey')
                  | (df_coords.content == 'nectar'), 'content_simple'] = 'nectar_honey'

    # comb height (y)
    df_coords['comb_ymin'] = df_coords.groupby(
        ['colony', 'week', 'z'])['y'].transform('min')
    df_coords['comb_ymax'] = df_coords.groupby(
        ['colony', 'week', 'z'])['y'].transform('max')
    df_coords['comb_height'] = df_coords['comb_ymax'] - df_coords['comb_ymin']
    df_coords['y_prop'] = ((df_coords['y'] - df_coords['comb_ymin'])
                           / df_coords['comb_height']) * 100

    # comb width (x)
    df_coords['comb_xmin'] = df_coords.groupby(
        ['colony', 'week', 'z'])['x'].transform('min')
    df_coords['comb_xmax'] = df_coords.groupby(
        ['colony', 'week', 'z'])['x'].transform('max')
    df_coords['comb_width'] = df_coords['comb_xmax'] - df_coords['comb_xmin']
    df_coords['x_prop'] = ((df_coords['x'] - df_coords['comb_xmin'])
                           / df_coords['comb_width']) * 100

    # reorder frames (z) so low values are closer to the nest initiation point
    df_coords['z_RO'] = 19 - df_coords['z']
    df_coords.loc[df_coords['colony'].isin(['DD6', 'DD7', 'SH4']), 'z_RO'] = \
        df_coords['z']

    # reorder x so low values are closer to the comb initiation side
    width = Contents[0][0][0].shape[1]
    df_coords['x_RO'] = df_coords['x']
    df_coords['comb_xmin_RO'] = df_coords['comb_xmin']
    df_coords.loc[(df_coords['comb_ymax'] == df_coords['y']),
                  'y_max_match'] = 1
    df_coords.loc[(df_coords['y_max_match'] == 1)
                  & (df_coords['x'] > width / 2), 'x_flip'] = 1
    df_coords['x_flip'] = df_coords.groupby(
        ['colony', 'week', 'z'])['x_flip'].transform('max')
    df_coords.loc[(df_coords['x_flip'] == 1), 'x_RO'] = width - df_coords['x']
    df_coords.loc[(df_coords['x_flip'] == 1), 'comb_xmin_RO'] = \
        width - df_coords['comb_xmin'] - df_coords['comb_width']
    df_coords['x_RO_prop'] = ((df_coords['x_RO'] - df_coords['comb_xmin_RO'])
                              / df_coords['comb_width']) * 100

    return df_coords


def figure_3_vertical_distribution(df_coords: pd.DataFrame,
                                   ContentsCrop: np.ndarray,
                                   treatment: str = 'DD', nbins: int = 4,
                                   out: str | None = None):
    """Figure 3B — vertical distribution of simplified contents by comb size.

    Comb-cell rows split into ``nbins`` power-spaced size classes
    (fraction of a full comb); filled density polygons of y-position per
    simplified content (pollen in the foreground).  Brood below, pollen
    middle, nectar/honey top.
    """
    content_simple = ['any_brood', 'pollen', 'nectar_honey']
    hexes = [simplehex[1], simplehex[0], simplehex[2]]
    maxsize = np.size(ContentsCrop[0][0][0])
    bins = util.powspace(start=0, stop=maxsize, power=1.5, num=nbins + 1)

    fig, ax = plt.subplots(1, nbins, figsize=(7, 4), sharey=True)
    ax[0].invert_yaxis()
    for i in range(nbins):
        lo = bins[i]
        hi = bins[i + 1] - 1
        data = df_coords[(df_coords['treatment'] == treatment)
                         & (df_coords['totalcomb'] > lo)
                         & (df_coords['totalcomb'] < hi)
                         & (df_coords['content_simple'].isin(content_simple))]
        data = pd.concat([data[data['content_simple'] == 'pollen'],
                          data[data['content_simple'] != 'pollen']])
        sns.histplot(ax=ax[i], data=data, y='y', hue='content_simple',
                     fill=True, alpha=.8, palette=hexes, element='poly',
                     binwidth=1)
        ax[i].set_xticks([])
        ax[i].set_xlabel(str(round(lo / maxsize, 2)) + ' - '
                         + str(round(hi / maxsize, 2)))
        ax[i].set_yticks([])
        sns.despine()
        if i == 0:
            ax[i].set_ylabel('Vertical distribution')
            sns.move_legend(ax[i], "lower right", title='',
                            labels=['pollen', 'any brood',
                                    'nectar &\nhoney'], frameon=False,
                            bbox_to_anchor=(1.2, 0), fontsize=10)
        else:
            ax[i].set_ylabel('')
            ax[i].legend([], [], frameon=False)

    fig.text(0.5, 0.03, 'Comb size (fraction of full comb)', ha='center')
    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax
