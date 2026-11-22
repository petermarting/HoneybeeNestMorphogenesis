"""Rebuild selected derived tables from the released comb masks.

The generators create the cell neighbourhood, cell future-self, and
comb-face nearest-neighbour tables. The nest-level generator also needs
the per-frame landscape-metrics table ``df_mergedlandscape.csv``.

Figures use the released CSVs. The table tests compare rebuilt output
with those reference inputs.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import spatial

from .constants import colony_names, content_names

#: Eight clockwise neighbour offsets from a focal cell, starting at noon.
NEIB_POSITIONS = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0],
                           [-1, -1], [0, -1], [1, -1]])

#: Contents tracked as focal types in the neighbourhood table (eggs..honey).
_NEIB_FOCUS = [3, 4, 5, 6, 7, 8]
#: Contents tracked in the future-self table (empty_comb..honey).
_FUTURE_FOCUS = [2, 3, 4, 5, 6, 7, 8]

_CLASS_NAMES = content_names  # 12 detailed content classes


def _add_trueweek(df: pd.DataFrame, path: str | None = None) -> pd.DataFrame:
    """Merge colony/week → (date, true_week) and reorder columns (Make cells 10/17)."""
    root = os.path.dirname(os.path.abspath(__file__)) + "/../../"
    df_trueweek = pd.read_csv(os.path.join(path or root, "df_trueweek.csv"))
    df = pd.merge(df, df_trueweek, how="left",
                  on=["colony", "week"]).reset_index(drop=True)
    cols = df.columns.tolist()
    return df[cols[:1] + ['date', 'true_week'] + cols[1:-2]]


def _add_phase(df: pd.DataFrame) -> pd.DataFrame:
    """seasonal_phase and treatment columns (Make cells 9/16)."""
    df['seasonal_phase'] = 'spring_growth'
    df.loc[(df.week > 11) & (df.week < 17), 'seasonal_phase'] = 'summer_dearth'
    df.loc[df.week > 16, 'seasonal_phase'] = 'fall_flow'
    df['treatment'] = df['colony'].str[:2]
    return df


def cell_neighbour_table(Contents: np.ndarray) -> pd.DataFrame:
    """``df_cellneib`` — neighbourhood composition of every focal cell (Make cells 7–10).

    For each comb face (side labels alternate ``a``/``b``) records, per
    focal content, the percentage of its eight immediate neighbours of
    each content class, plus the same percentage over the whole frame for
    contrast (background/wood on-frame percentages set to NaN;
    ``perc_diff`` NaN where the frame percentage is 0).  Border focal
    cells (last row/column) are skipped, as in the notebook.  ~1 hour.

    NOTE: ``frame_reoriented`` uses the notebook's always-true
    ``(colony!='DD6')|(colony!='DD7')|(colony!='SH4')`` condition, so it
    equals ``11 - frame`` for every colony — the committed CSV has exactly
    that (the intended DD6/DD7/SH4 forward order lives in
    ``df_mergedinsp['frame_RO']``).
    """
    side = 'a'
    frames = []
    for c in range(len(Contents)):
        print(colony_names[c], flush=True)
        for w in range(len(Contents[c])):
            for f in range(20):
                mask = Contents[c][w][f]
                ncells = np.count_nonzero(mask > 1)

                for focalcontent in _NEIB_FOCUS:
                    coords = np.argwhere(mask == focalcontent)
                    if len(coords) == 0:
                        continue

                    neiblistall = []
                    for focalpos in coords:
                        if ((focalpos[0] < mask.shape[0] - 1)
                                and (focalpos[1] < mask.shape[1] - 1)):
                            neiblist = []
                            for neibposition in NEIB_POSITIONS:
                                combpos = focalpos + neibposition
                                neiblist.append(mask[combpos[0]][combpos[1]])
                            neiblistall.append(neiblist)
                    neiblistall = np.array(neiblistall)

                    neibsum = len(neiblistall) * len(neiblist)
                    for content in range(len(_CLASS_NAMES)):
                        contentcount = np.count_nonzero(
                            neiblistall == content)
                        contentperc = (contentcount / neibsum) * 100
                        frames.append({
                            'colony': colony_names[c],
                            'week': w + 1,
                            'frame': int(f / 2) + 1,
                            'side': side,
                            'framenum': f,
                            'focalcontent': _CLASS_NAMES[focalcontent],
                            'nfocalcells': len(coords),
                            'neibcontent': _CLASS_NAMES[content],
                            'percniebcontent_onframe':
                                (np.count_nonzero(mask == content) / ncells) * 100,
                            'neibperc': contentperc,
                        })

                side = 'b' if side == 'a' else 'a'
    df = pd.DataFrame(frames)

    df.loc[(df.neibcontent == 'background')
           | (df.neibcontent == 'wood'), 'percniebcontent_onframe'] = np.nan
    df['perc_diff'] = df.neibperc - df.percniebcontent_onframe
    df.loc[df.percniebcontent_onframe == 0, 'perc_diff'] = np.nan

    df['frame_reoriented'] = 11 - df['frame']
    df = _add_phase(df)
    return _add_trueweek(df)


def cell_future_table(Contents: np.ndarray) -> pd.DataFrame:
    """``df_cellfuture`` — week-to-week self-transitions of every cell (Make cells 15–18).

    Each focal cell of each frame is looked up at the same position in
    the next week's mask; records, per focal content, the percentage that
    became each content one week later.  ~25 min.  (Same
    ``frame_reoriented`` NOTE as :func:`cell_neighbour_table`.)
    """
    side = 'a'
    frames = []
    for c in range(len(Contents)):
        print(colony_names[c], flush=True)
        for w in range(len(Contents[c]) - 1):
            for f in range(20):
                mask0 = Contents[c][w][f]
                mask1 = Contents[c][w + 1][f]

                for focalcontent in _FUTURE_FOCUS:
                    coords = np.argwhere(mask0 == focalcontent)
                    if len(coords) == 0:
                        continue
                    selflist = mask1[coords[:, 0], coords[:, 1]]

                    for content in _FUTURE_FOCUS:
                        contentcount = np.count_nonzero(selflist == content)
                        contentperc = (contentcount / len(selflist)) * 100
                        frames.append({
                            'colony': colony_names[c],
                            'week': w + 1,
                            'frame': int(f / 2) + 1,
                            'side': side,
                            'framenum': f,
                            'focalcontent': _CLASS_NAMES[focalcontent],
                            'nfocalcells': len(coords),
                            'nextself': _CLASS_NAMES[content],
                            'nextselfcount': contentcount,
                            'nextperc': contentperc,
                        })

                side = 'b' if side == 'a' else 'a'
    df = pd.DataFrame(frames)
    df['frame_reoriented'] = 11 - df['frame']
    df = _add_phase(df)
    return _add_trueweek(df)


def _weighted_average(dataframe, value, weight):
    val = dataframe[value]
    wt = dataframe[weight]
    return (val * wt).sum() / wt.sum()


def nearest_neighbour_table(ContentsCrop: np.ndarray, thres: int = 10,
                            nsamples: int = 100,
                            seed: int | None = 42):
    """``df_nearestdist`` (+ ``_sum``) — comb-face cluster metric (Make cells 22–27).

    Per comb face and content (> ``thres`` cells): mean distance to the
    nearest same-content neighbour vs the mean over ``nsamples`` random
    relocations of that content on the comb (averaged), the Monte-Carlo
    p-value ``(#rand <= obs + 1)/(n+1)``, and the cluster metric
    ``1 - (obs-1)/(rand-1)`` (0 = random, 1 = maximally clustered).
    ``_sum`` weights the metric by cell count per frame. The seed makes
    the sampling repeatable. The nearest-neighbour test compares
    deterministic fields and Monte Carlo statistics using their
    respective tolerances.
    """
    if seed is not None:
        np.random.seed(seed)
    side = 'a'
    appended = []
    # NOTE(original): the notebook hardcoded range(12) colonies; iterating
    # len(ContentsCrop) is identical for the full 12-colony array and
    # permits subset re-runs for validation.
    for c in range(len(ContentsCrop)):
        for w in range(len(ContentsCrop[c])):
            for f in range(20):
                mask = ContentsCrop[c][w][f]
                combcoords = np.argwhere(mask > 1)

                for content in range(3, 9):
                    contentcoords = np.argwhere(mask == content)
                    if len(contentcoords) <= thres:
                        continue

                    randlist = []
                    for _n in range(nsamples):
                        randindex = np.random.choice(
                            len(combcoords), len(contentcoords),
                            replace=False)
                        randcoords = combcoords[randindex]
                        randtree = spatial.cKDTree(randcoords)
                        rd, _ri = randtree.query(randcoords, k=2)
                        randlist.append(np.mean(rd[:, 1]))
                    randnearestmean = np.mean(randlist)

                    tree = spatial.cKDTree(contentcoords)
                    d, _i = tree.query(contentcoords, k=2)
                    nearestmean = np.mean(d[:, 1])

                    p_value = (np.sum(np.array(randlist) <= nearestmean)
                               + 1) / (nsamples + 1)

                    appended.append({
                        'colony': colony_names[c],
                        'week': w + 1,
                        'frame': int(f / 2) + 1,
                        'side': side,
                        'treatment': colony_names[c][:2],
                        'contentid': content,
                        'content': content_names[content],
                        'count': len(contentcoords),
                        'rand_nearestneibavg': randnearestmean,
                        'nearestneibavg': nearestmean,
                        'combcount': len(combcoords),
                        'p_value': p_value,
                    })

                side = 'b' if side == 'a' else 'a'
            print(f'colony: {colony_names[c]}, week: {str(w + 1).zfill(2)}',
                  end='\r', flush=True)

    df = pd.DataFrame(appended)
    df['content_density'] = df['count'] / df['combcount']
    df['diff'] = df['rand_nearestneibavg'] - df['nearestneibavg']
    df['perc_diff'] = round((df['diff'] / df['rand_nearestneibavg']) * 100, 2)
    df['dispersion_metric'] = 1 - ((df['nearestneibavg'] - 1)
                                   / (df['rand_nearestneibavg'] - 1))

    df_sum = pd.DataFrame(df.groupby(
        ['colony', 'treatment', 'week', 'frame', 'side']).apply(
            _weighted_average, 'dispersion_metric', 'count')).reset_index()
    df_sum.rename(columns={0: 'dispersion_metric'}, inplace=True)

    return _add_trueweek(df), _add_trueweek(df_sum)


def nest_nearest_neighbour_table(ContentsCrop: np.ndarray,
                                 df_mergedlandscape: pd.DataFrame,
                                 thres: int = 50, nsamples: int = 100):
    """``df_nearestdist_nest`` (+ ``_sum``) — 3D nest-level cluster metric.

    Requires ``df_mergedlandscape.csv`` (the per-frame landscape table,
    **not released** with the paper) to arrange each nest, and is
    additionally unseeded in the original notebook — so the committed
    nest tables cannot be reproduced bit-for-bit; treat them as
    canonical (see ``RESULTS.md``).  This function ports the computation
    (Make cells 31–33) for completeness once the landscape table is
    available.
    """
    from .util import arrange_nest

    appended = []
    for c in range(12):
        for w in range(len(ContentsCrop[c])):
            masks = ContentsCrop[c][w]
            masks = arrange_nest(masks, colony_names[c], w,
                                 df_mergedlandscape)
            combcoords = np.argwhere(masks > 1)

            for content in range(3, 9):
                contentcoords = np.argwhere(masks == content)
                if len(contentcoords) <= thres:
                    continue

                randlist = []
                for _n in range(nsamples):
                    randindex = np.random.choice(
                        len(combcoords), len(contentcoords), replace=False)
                    randcoords = combcoords[randindex]
                    randtree = spatial.cKDTree(randcoords)
                    rd, _ri = randtree.query(randcoords, k=2)
                    randlist.append(np.mean(rd[:, 1]))
                randnearestmean = np.mean(randlist)

                tree = spatial.cKDTree(contentcoords)
                d, _i = tree.query(contentcoords, k=2)
                nearestmean = np.mean(d[:, 1])

                p_value = (np.sum(np.array(randlist) <= nearestmean)
                           + 1) / (nsamples + 1)

                appended.append({
                    'colony': colony_names[c],
                    'week': w + 1,
                    'treatment': colony_names[c][:2],
                    'contentid': content,
                    'content': content_names[content],
                    'count': len(contentcoords),
                    'rand_nearestneibavg': randnearestmean,
                    'nearestneibavg': nearestmean,
                    'combcount': len(combcoords),
                    'p_value': p_value,
                })

    df = pd.DataFrame(appended)
    df['content_density'] = df['count'] / df['combcount']
    df['diff'] = df['rand_nearestneibavg'] - df['nearestneibavg']
    df['perc_diff'] = round((df['diff'] / df['rand_nearestneibavg']) * 100, 2)
    df['dispersion_metric'] = 1 - ((df['nearestneibavg'] - 1)
                                   / (df['rand_nearestneibavg'] - 1))

    df_sum = pd.DataFrame(df.groupby(
        ['colony', 'treatment', 'week']).apply(
            _weighted_average, 'dispersion_metric', 'count')).reset_index()
    df_sum.rename(columns={0: 'dispersion_metric'}, inplace=True)
    return df, df_sum
