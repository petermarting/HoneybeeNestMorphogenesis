"""Loaders for the comb-content masks and statistics tables.

The masks (``CombMasks_*.npy``) and derived tables (``df_*.csv``) live in the
repository root; they were produced by the paper's imaging pipeline
(see ``RESULTS.md`` for provenance and the one table whose generator is not
released).  The loading/derivation here reproduces the "Import masks" and
"Import dataframes" cells of ``CombContent_Analysis.ipynb`` verbatim, so the
returned frames are identical to those the original notebook analysed.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .constants import within_frame_width_cm

#: Mask arrays and their ``.npy`` file names.
MASK_FILES = {
    "Contents": "CombMasks_Contents_aligned.npy",
    "ContentsCrop": "CombMasks_Contents_crop.npy",
    "ContentsSimple": "CombMasks_Contents_simple.npy",
    "CombType": "CombMasks_CombTypes_aligned.npy",
    "Combined": "CombMasks_Combined_aligned.npy",
}

#: Table files as loaded (before the derived-column/merge steps below).
TABLE_FILES = {
    "df_combcounts": "df_combcounts.csv",
    "df_cellneib": "df_cellneib.csv",
    "df_cellfuture": "df_cellfuture.csv",
    "df_nearestdist": "df_nearestdist.csv",
    "df_nearestdist_sum": "df_nearestdist_sum.csv",
    "df_nearestdist_nest": "df_nearestdist_nest.csv",
    "df_nearestdist_nest_sum": "df_nearestdist_nest_sum.csv",
    "df_trueweek": "df_trueweek.csv",
}


def _root(path: str | os.PathLike | None = None) -> str:
    """Locate the directory holding the mask/table files.

    Defaults to the current working directory; if the data is not there,
    walks up from this package's location (the repository root).
    """
    if path is not None:
        return str(path)
    for d in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        cur = d
        while True:
            if os.path.exists(os.path.join(cur, "df_combcounts.csv")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    return os.getcwd()


def load_masks(path: str | os.PathLike | None = None,
               keys: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Load comb mask arrays as a dict keyed by ``MASK_FILES`` names.

    Arrays are ragged object arrays (one entry per colony, each a
    ``(week, frame_side)`` collection of 2D label masks), hence
    ``allow_pickle=True``.  Index like ``Contents[colony][week][frame_side]``
    with ``colony`` the row index in ``constants.colony_names``.
    """
    root = _root(path)
    keys = tuple(MASK_FILES) if keys is None else keys
    out = {}
    for k in keys:
        out[k] = np.load(os.path.join(root, MASK_FILES[k]), allow_pickle=True)
    return out


def px_to_cm(ContentsCrop: np.ndarray) -> float:
    """Pixel-to-centimetre factor from the frame-interior crop.

    Same derivation as the notebook: the cropped comb width is
    ``constants.within_frame_width_cm`` centimetres.
    """
    width_px = ContentsCrop[0][0][0].shape[1]
    return within_frame_width_cm / width_px


def load_tables(path: str | os.PathLike | None = None,
                merge_dispersion: bool = True) -> dict[str, pd.DataFrame]:
    """Load the statistics tables, reproducing the notebook's derived columns.

    Returns a dict with the loaded frames (keyed as ``TABLE_FILES``) plus
    ``"df_mergedinsp"``: the comb-count table with the notebook's derived
    columns -- percentage of bees on frame, merged nectar+honey, comb
    perimeter/area ratio, ``frame_RO`` (frame order from the nest-initiation
    wall; forward for DD6/DD7/SH4, reversed otherwise) -- and, when
    ``merge_dispersion`` (the default), the left-join of the comb- and
    nest-level dispersion (nearest-neighbour) summaries into it, exactly as
    the "Import dataframes" cell does.
    """
    root = _root(path)
    tables = {k: pd.read_csv(os.path.join(root, fn))
              for k, fn in TABLE_FILES.items()}
    df = tables["df_combcounts"]

    # Derived columns (verbatim from the notebook).
    df['perc_bees_onframe'] = df['bees_onframe'] / df['bees_sum_perweek']
    df['nectar_honey'] = df[['nectar', 'honey']].sum(axis=1)
    df['perc_nectar_honey'] = (df['nectar_honey'] / df['totalcomb']) * 100
    df['comb_perim_area_ratio'] = df['comb_perim'] / np.sqrt(df['totalcomb'])

    df['frame_RO'] = 11 - df['frame']
    df.loc[(df['colony'] == 'DD6') | (df['colony'] == 'DD7')
           | (df['colony'] == 'SH4'), 'frame_RO'] = df['frame']

    if merge_dispersion:
        df = df.merge(tables["df_nearestdist_sum"], how='left',
                      on=['colony', 'treatment', 'date', 'week', 'true_week',
                          'frame', 'side'])
        df = df.merge(tables["df_nearestdist_nest_sum"], how='left',
                      on=['colony', 'treatment', 'date', 'week', 'true_week'])
        df = df.rename(columns={'dispersion_metric_x': 'dispersion_metric',
                                'dispersion_metric_y': 'dispersion_metric_nest'})

    tables["df_mergedinsp"] = df
    return tables
