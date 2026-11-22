"""Rebuild the derived comb-content tables from the comb masks.

Re-runs the slow generators from ``combinator.content.make`` (the generators
from the original ``CombContent_Make.ipynb``) over the released mask arrays.  Usage (from the package root):

    uv run python -m combinator.make_content_tables \
        --tables nearestdist --outdir tables/

Choices for ``--tables``: ``cellneib`` (~1 h), ``cellfuture`` (~25 min),
``nearestdist`` (~10 min, seeded reproducible).  Outputs land in
``--outdir`` (default: the repository root, overwriting nothing unless
``--overwrite`` is passed).

The nest-level tables (``df_nearestdist_nest*.csv``) are **not**
regenerable: their generator needs the unreleased
``df_mergedlandscape.csv`` (see ``RESULTS.md``); the committed versions
are canonical.
"""

from __future__ import annotations

import argparse
import os

from .content import data, make


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tables", default="cellneib,cellfuture,nearestdist",
                    help="comma-separated: cellneib,cellfuture,nearestdist")
    ap.add_argument("--root", default=None,
                    help="directory holding CombMasks_*.npy (default: auto)")
    ap.add_argument("--outdir", default=None,
                    help="where to write CSVs (default: --root)")
    ap.add_argument("--seed", type=int, default=42,
                    help="numpy seed for nearestdist sampling "
                         "(notebook default 42; -1 = unseeded)")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing CSVs in --outdir")
    args = ap.parse_args()

    root = data._root(args.root)
    outdir = args.outdir or root
    os.makedirs(outdir, exist_ok=True)

    def save(df, name):
        path = os.path.join(outdir, name)
        if os.path.exists(path) and not args.overwrite:
            raise SystemExit(f"{path} exists; pass --overwrite to replace")
        df.to_csv(path, index=False)
        print(f"wrote {path} ({df.shape[0]} rows)")

    tables = {t.strip() for t in args.tables.split(",")}
    unknown = tables - {"cellneib", "cellfuture", "nearestdist"}
    if unknown:
        raise SystemExit(f"unknown table(s): {', '.join(sorted(unknown))}")

    if "nearestdist" in tables:
        masks = data.load_masks(root, keys=("ContentsCrop",))
        df, df_sum = make.nearest_neighbour_table(
            masks["ContentsCrop"],
            seed=None if args.seed < 0 else args.seed)
        save(df, "df_nearestdist.csv")
        save(df_sum, "df_nearestdist_sum.csv")

    if {"cellneib", "cellfuture"} & tables:
        masks = data.load_masks(root, keys=("Contents",))
        if "cellneib" in tables:
            save(make.cell_neighbour_table(masks["Contents"]),
                 "df_cellneib.csv")
        if "cellfuture" in tables:
            save(make.cell_future_table(masks["Contents"]),
                 "df_cellfuture.csv")


if __name__ == "__main__":
    main()
