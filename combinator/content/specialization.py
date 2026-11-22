"""Cell and comb specialization (paper Figure 4B, D, E, F).

Derived from the paper's ``CombContent_Analysis.ipynb``.

Panels (legend, Figure 4): (A) schematic of the cell-sequence concept —
drawn, no code; (B) ten-week content histories of every cell in all
unshuffled nests, sorted by specialization (``plot_cell_specialists``);
(C) 2D kernel densities of specialist positions on the comb face — **no
released code** (see ``RESULTS.md``); (D) vertical / horizontal /
across-comb density distributions of specialist cells; (E) observed vs
randomized-nest specialization rates (paired t tests via
``add_sig_note``); (F) comb-level specialization grid (colony × comb
position).
"""

from __future__ import annotations

import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

from . import util
from .constants import (colony_names, contenthex, xmincrop, xmaxcrop,
                        ymincrop, ymaxcrop)

#: x-axis labels for the specialization-comparison figure (Figure 4E).
_SPECIALIST_LABELS = ['generalists', 'empty', 'brood', 'pollen', 'nectar']


def cell_specialization(Contents: np.ndarray, maxweek: int = 10):
    """Cell-level specialization of the six unshuffled colonies (Figure 4B).

    Returns the ``util.cell_specialists`` result object (sequences,
    proportions, mean+1SD specialist thresholds, classifications).
    Takes ~12 s.
    """
    return util.cell_specialists(Contents[:6], maxweek=maxweek,
                                 prop_thres=None)


def figure_4_cell_sequences(Contents: np.ndarray, maxweek: int = 10,
                            figsize: tuple = (4, 7.5), out: str | None = None):
    """Figure 4B — sorted cell content-history sequences for all DD nests."""
    cell_sp = cell_specialization(Contents, maxweek=maxweek)
    # NOTE: the original plot_cell_specialists has no return statement
    # (notebook style); grab the figure it built via the pyplot state.
    util.plot_cell_specialists(cell_sp, maxweek=maxweek, figsize=figsize,
                               contents=Contents)
    fig = plt.gcf()
    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig


def stratification_sequences(Contents: np.ndarray, maxweek: int = 10):
    """Cell sequences with comb-position bookkeeping for Figure 4D.

    For every comb face of the six unshuffled colonies (both sides, side
    b mirrored so both faces share an orientation), records the simplified
    10-week content sequence and the cell's (x, y, comb) coordinates;
    comb index ``z`` is re-oriented so low values are at the nest
    initiation wall (forward for DD6/DD7).  Thresholds and specialist
    classes are the mean+1SD rule over all DD cells.  Returns a namespace
    with ``ids, classification_names, pts_sep, z_sep`` (per class).
    """
    slist, ptlist, zlist = [], [], []
    for c in range(6):
        masks0 = Contents[c][:maxweek]

        # filter out points outside of frame
        masks = np.copy(masks0)
        masks[:,:,:ymincrop,:] = 0
        masks[:,:,ymaxcrop:,:] = 0
        masks[:,:,:,:xmincrop] = 0
        masks[:,:,:,xmaxcrop:] = 0

        for f in range(10):
            for side in range(2):  # both sides of comb
                mask = masks[:maxweek, f*2+side]
                if side == 1:
                    mask = np.flip(mask, axis=2)
                s, pts = util.get_cell_sequences(mask)
                s = np.where((s > 2) & (s < 6), 5, s)   # brood -> one value
                s = np.where((s == 7) | (s == 8), 7, s)  # nectar+honey
                s = np.where(s > 8, 2, s)                # qcup/bees -> empty
                slist.extend(s)
                ptlist.extend(pts)
                if c > 3:
                    zlist.extend([f] * len(s))
                else:
                    zlist.extend([9 - f] * len(s))

    slist = np.array(slist)
    pts = np.array(ptlist)
    zlist = np.array(zlist)

    counts, proportions = util.get_counts_and_proportions(slist)

    # specialist thresholds: mean + 1 SD over all DD cells
    prop_means = np.mean(proportions, axis=0)
    prop_std = np.std(proportions, axis=0)
    prop_thres = prop_means + prop_std

    classification_names = ['generalists', 'empty specialists',
                            'brood specialists', 'pollen specialists',
                            'nectar specialists']
    classifications = []
    for seq in proportions:
        if seq[0] > prop_thres[0]:
            id = 2   # empty specialist
        elif seq[1] > prop_thres[1]:
            id = 5   # brood specialist
        elif seq[2] > prop_thres[2]:
            id = 6   # pollen specialist
        elif seq[3] > prop_thres[3]:
            id = 7   # nectar specialist
        else:
            id = 0   # generalist / mixed use
        classifications.append(id)
    classifications = np.array(classifications)

    ids, idcounts = np.unique(classifications, return_counts=True)
    class_props = (idcounts / len(slist)) * 100

    # separate positions by class
    pts_sep, z_sep = [], []
    for i in ids:
        idx = np.argwhere(classifications == i)
        pts_sep.append(np.squeeze(pts[idx]))
        z_sep.append(np.squeeze(zlist[idx]))

    out_ns = types.SimpleNamespace(
        ids=ids, classification_names=classification_names,
        class_props=class_props, prop_thres=prop_thres,
        pts_sep=pts_sep, z_sep=z_sep)
    return out_ns


def figure_4_stratification(strat: types.SimpleNamespace,
                            Contents: np.ndarray, px_to_cm_factor: float,
                            out: str | None = None):
    """Figure 4D — specialist densities in vertical, horizontal, across-comb position."""
    plt.style.use('default')
    fig = plt.figure(figsize=(5, 2.75), constrained_layout=False)

    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1],
                  width_ratios=[1, 1.9], wspace=.2, hspace=.2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    for i in range(len(strat.pts_sep)):
        points = strat.pts_sep[i]
        x = points[:, 0]
        y = Contents[0][0][0].shape[0] - points[:, 1]
        z = strat.z_sep[i] + 1

        sns.histplot(ax=ax1, y=y * px_to_cm_factor, fill=True, alpha=.3,
                     color=contenthex[strat.ids[i]],
                     label=strat.classification_names[i], stat='density',
                     element='poly', binwidth=1)
        sns.histplot(ax=ax2, x=x * px_to_cm_factor, fill=True, alpha=.3,
                     color=contenthex[strat.ids[i]],
                     label=strat.classification_names[i], stat='density',
                     element='poly', binwidth=1)
        sns.histplot(ax=ax3, x=z, fill=True, alpha=.3,
                     color=contenthex[strat.ids[i]],
                     label=strat.classification_names[i], stat='density',
                     element='poly', binwidth=1)

    ax1.set_xticks([])
    ax1.set_yticklabels((ax1.get_yticks()).astype(int), fontsize=8)
    ax1.set_ylabel('Vertical position (cm)')
    handles, labels = ax1.get_legend_handles_labels()
    ax3.legend(handles[::-1], labels[::-1], loc="upper right",
               bbox_to_anchor=(.9, 1), frameon=False, fontsize=9)

    ax2.set_xticklabels((ax2.get_xticks()).astype(int), fontsize=8)
    ax2.set_yticks([])
    ax2.set_xlabel('Horizontal position (cm)')

    ax3.set_xticklabels((ax3.get_xticks()).astype(int), fontsize=8)
    ax3.set_yticks([])
    ax3.set_xlabel('Position across combs')
    sns.despine()

    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig


def cell_specialization_comparison(Contents: np.ndarray, maxweek: int = 10):
    """Observed vs randomized-nest specialization proportions (Figure 4E).

    Thresholds are computed once on all DD cells; then each colony's
    observed and cell-shuffled (``randomize_cells_within_nests``)
    specialization spectrum is measured with those fixed thresholds.
    Returns ``df_class_props`` (colony × observed/randomized × class).
    Takes ~30 s.
    """
    specialists = util.cell_specialists(Contents[:6], maxweek=maxweek,
                                        prop_thres=None)
    prop_thres = specialists.prop_thres

    df_class_props = []
    for c in range(6):
        masks = Contents[c]

        specialists = util.cell_specialists(masks, maxweek=maxweek,
                                            prop_thres=prop_thres,
                                            print_results=False)
        df_temp = pd.DataFrame()
        df_temp['colony'] = [colony_names[c]] * len(specialists.ids)
        df_temp['treatment'] = ['observed'] * len(specialists.ids)
        df_temp['specialization'] = specialists.classification_names
        df_temp['class_ids'] = specialists.ids
        df_temp['class_props'] = specialists.class_props
        df_class_props.append(df_temp)

        masks_rand = util.randomize_cells_within_nests(masks,
                                                       maxweek=maxweek)
        specialists = util.cell_specialists(masks_rand, maxweek=maxweek,
                                            prop_thres=prop_thres,
                                            print_results=False)
        df_temp = pd.DataFrame()
        df_temp['colony'] = [colony_names[c]] * len(specialists.ids)
        df_temp['treatment'] = ['randomized'] * len(specialists.ids)
        df_temp['specialization'] = specialists.classification_names
        df_temp['class_ids'] = specialists.ids
        df_temp['class_props'] = specialists.class_props
        df_class_props.append(df_temp)

    return pd.concat(df_class_props, ignore_index=True)


def figure_4_specialization_vs_randomized(df_class_props: pd.DataFrame,
                                          out: str | None = None):
    """Figure 4E — specialization rates, observed vs randomized nests.

    Stars from :func:`util.add_sig_note` with ``method='paired'``
    (paired t tests, matching the figure legend).
    """
    x, y, hue = 'specialization', 'class_props', 'treatment'

    fig, ax0 = plt.subplots(figsize=(4, 2))
    sns.stripplot(ax=ax0, data=df_class_props, x=x, y=y, hue=hue,
                  palette=['k', 'grey'], dodge=True, alpha=0.9)
    sns.boxplot(ax=ax0, data=df_class_props, x=x, y=y, hue=hue,
                palette=['w', 'w'], showfliers=False, legend=False)
    sns.despine()
    util.add_sig_note(ax0, df_class_props, x, y, hue, y_offset=2,
                      method='paired')
    ax0.legend(frameon=False, handletextpad=-.2)
    ax0.set_ylabel('% of cells in the nest')
    ax0.set_xlabel('')
    ax0.set_xticks(ticks=range(len(_SPECIALIST_LABELS)),
                   labels=_SPECIALIST_LABELS, rotation=30)
    ax0.set_ylim(0)

    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax0


def comb_specialization(Contents: np.ndarray, maxweek: int = 10):
    """Comb-level specialization of the DD colonies (Figure 4F computation).

    NOTE: the per-class comb percentages quoted in the paper text
    ("56.7% ± 17.0% of combs specialized ... nectar 17.5% ± 5.0%; pollen
    16.7% ± 8.2% ...") cannot be recomputed exactly from the released
    pipeline — pollen matches to the digit (16.7 ± 8.2) but the other
    classes differ by a few points under every denominator tried; the
    aggregation used for the text was evidently a variant of
    ``comb_specialists`` not published (documented gap).

    Cell thresholds from :func:`cell_specialization`; comb thresholds are
    computed per ``util.comb_specialists`` defaults (mean+1SD across
    combs).  Takes ~35 s.
    """
    cell_sp = cell_specialization(Contents, maxweek=maxweek)
    return util.comb_specialists(Contents[:6], maxweek=maxweek,
                                 cell_prop_thres=cell_sp.prop_thres,
                                 comb_prop_thres=None, print_results=True)


def figure_4_comb_specialization(comb_sp, out: str | None = None):
    """Figure 4F — comb specialization grid (colony × comb position).

    NOTE: the blank square at DD6 comb position 10 is hardcoded in the
    notebook — no comb was built on the 10th frame there during weeks
    1–10.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for c in range(len(comb_sp.comb_ids)):
        for f in range(len(comb_sp.comb_ids[c])):
            id = comb_sp.comb_ids[c][f]
            x = f + 1
            y = len(comb_sp.comb_ids) - (c + 1)
            if c == 4 and f == 9:
                color = 'w'
            else:
                color = contenthex[id]
            ax.scatter(x=x, y=y, c=color, s=300, marker='s')

    ax.set_xlabel('Comb position across nest')
    ax.set_ylabel('Colony ID')
    ax.set_yticks(np.arange(len(colony_names[:6])))
    ax.set_yticklabels(colony_names[:6][::-1])
    ax.set_ylim(-.6, 5.6)
    ax.legend(['nectar', 'pollen', 'brood', 'empty', 'generalist'],
              title='specialization', bbox_to_anchor=(1.5, 1),
              loc='upper right', frameon=False)
    leg = ax.get_legend()
    colors = [7, 6, 5, 2, 0]
    for i in range(len(colors)):
        leg.legend_handles[i].set_facecolor(contenthex[colors[i]])
        leg.legend_handles[i].set_edgecolor('w')
    sns.despine()

    if out is not None:
        fig.savefig(out, dpi=300, bbox_inches='tight')
    return fig, ax
