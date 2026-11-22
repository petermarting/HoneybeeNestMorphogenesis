"""Persist and reconstruct the full per-hive model-output ``results`` dict.

``evaluate_spatial_classifier`` (see ``combinator/eval.py``) returns a dict::

    {
        (k, d, h, w, pos, ctx, int): {          # one entry per model config
            "hives": [                        # 18 hives, in fixed order
                {"hive_id": int,
                 "ground_truth":   (weeks, 20, 52, 104) int,
                 "predictions":    (weeks, 20, 52, 104) int,
                 "entropy":        (weeks, 20, 52, 104) float,
                 "log_likelihood": (weeks, 20, 52, 104) float},
                ...
            ]
        }, ...
    }

Held in memory this is on the order of a few GB.  To keep it storable in a
plain git repository (no LFS; GitHub's per-file cap is 100 MB) it is written as
**one small ``.npz`` per (model config, hive)** -- the largest hive (20 weeks)
is well under 20 MB -- plus a JSON manifest listing every file, its shape, and
its SHA-256.  ``load_results`` reassembles exactly the dict above from that
directory, so downstream figure/reanalysis code is identical whether the data
came from a live GPU run or from the committed files.

Storage:
    - ``ground_truth`` / ``predictions`` : int16 (labels are 0..5)
    - ``entropy`` / ``log_likelihood``  : float16 (values are small; the
                                           committed figures are reproduced to
                                           plotting precision)

Usage (as a module)::

    python -m combinator.results_io save   RESULTS_DIR
    python -m combinator.results_io load   RESULTS_DIR   (prints shapes)

Note: the ``results/`` directory committed with this repository was written by
the pre-rename package and its ``manifest.json`` still carries the format
string ``smitheebe.results v1``; that string is not validated on load, so the
committed results verify fine.  New saves write ``combinator.results v1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# naming helpers
# ---------------------------------------------------------------------------

def key_to_name(key) -> str:
    """Map a model config tuple to a filesystem-safe name.

    ``(1, F, F, F, F, F, 0)   -> k1_dF_hF_wF_posF_ctxF_int0
    (3, T, T, T, T, T, 1)     -> k3_dT_hT_wT_posT_ctxT_int1
    """
    k, d, h, w, pos, ctx, inter = key

    def b(x):
        return "T" if x else "F"

    return f"k{k}_d{b(d)}_h{b(h)}_w{b(w)}_pos{b(pos)}_ctx{b(ctx)}_int{inter}"


# The fields held per hive and how they are stored on disk.
FIELDS = ("ground_truth", "predictions", "entropy", "log_likelihood")
INT_FIELDS = {"ground_truth", "predictions"}


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_results(results: dict, outdir: str) -> str:
    """Write ``results`` to ``outdir`` as per-(config, hive) ``.npz`` files
    plus ``manifest.json``.  Returns the manifest path."""
    os.makedirs(outdir, exist_ok=True)
    manifest = {"format": "combinator.results v1", "fields": list(FIELDS)}
    models = {}
    for key, out in results.items():
        name = key_to_name(key)
        hives = {}
        for hive in out["hives"]:
            hid = int(hive["hive_id"])
            fname = f"cfg_{name}_hive_{hid:02d}.npz"
            fpath = os.path.join(outdir, fname)
            payload = {"hive_id": np.array(int(hid), dtype=np.int64)}
            shapes = {}
            for field in FIELDS:
                arr = np.asarray(hive[field])
                if field in INT_FIELDS:
                    payload[field] = arr.astype(np.int16)
                    shapes[field] = [list(arr.shape), "int16",
                                     int(arr.min()), int(arr.max())]
                else:
                    payload[field] = arr.astype(np.float16)
                    shapes[field] = [list(arr.shape), "float16",
                                     float(arr.min()), float(arr.max())]
            np.savez_compressed(fpath, **payload)
            hives[str(hid)] = {
                "file": fname,
                "sha256": _sha256(fpath),
                "n_weeks": int(arr.shape[0]),
                "shapes": shapes,
            }
        models[name] = {"key": list(key), "hives": hives}

    manifest["models"] = models
    mpath = os.path.join(outdir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return mpath


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def load_results(results_dir: str, verify: bool = True) -> dict:
    """Reassemble the ``results`` dict from a directory written by
    :func:`save_results`.  Float fields are returned as ``float32`` and label
    fields as ``int64`` -- matching the dtype of a fresh
    ``evaluate_spatial_classifier`` call."""
    with open(os.path.join(results_dir, "manifest.json")) as f:
        manifest = json.load(f)

    results = {}
    for name, model in manifest["models"].items():
        key = tuple(model["key"])
        hives = []
        for hid in sorted(model["hives"], key=int):
            meta = model["hives"][hid]
            fname = meta["file"]
            fpath = os.path.join(results_dir, fname)
            if verify:
                actual = _sha256(fpath)
                if actual != meta["sha256"]:
                    raise RuntimeError(
                        f"SHA-256 mismatch for {fname}: {actual} != "
                        f"{meta['sha256']} (file corrupted)")
            with np.load(fpath) as data:
                hive = {"hive_id": int(data["hive_id"])}
                for field in FIELDS:
                    arr = data[field]
                    hive[field] = (arr.astype(np.int64) if field in INT_FIELDS
                                   else arr.astype(np.float32))
            hives.append(hive)
        # preserve original hive ordering (by hive_id)
        hives.sort(key=lambda hv: hv["hive_id"])
        results[key] = {"hives": hives}
    return results


def verify_dir(results_dir: str) -> int:
    """Check the SHA-256 of every file against the manifest; returns file count."""
    with open(os.path.join(results_dir, "manifest.json")) as f:
        manifest = json.load(f)
    n = 0
    for model in manifest["models"].values():
        for meta in model["hives"].values():
            fpath = os.path.join(results_dir, meta["file"])
            if _sha256(fpath) != meta["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch for {meta['file']}")
            n += 1
    return n


def from_blob(blob: dict) -> dict:
    """Turn the flat ``"{key}{hive}_{field}"`` blob written by
    ``make_figures._save_results`` into the nested results dict."""
    import re
    grouped = {}
    for key, val in blob.items():
        m = re.match(r"^\(([^)]+)\)(\d+)_(\w+)$", key)
        if not m:
            continue
        parts = [x.strip() for x in m.group(1).split(",")]
        tup = (int(parts[0]),
               parts[1] == "True", parts[2] == "True", parts[3] == "True",
               parts[4] == "True", parts[5] == "True", int(parts[6]))
        hid = int(m.group(2))
        field = m.group(3)
        grouped.setdefault(tup, {}).setdefault(hid, {})[field] = val
    out = {}
    for tup, per_hive in grouped.items():
        hives = []
        for hid in sorted(per_hive):
            d = per_hive[hid]
            hives.append({
                "hive_id": hid,
                "ground_truth": d["ground_truth"],
                "predictions": d["predictions"],
                "entropy": d["entropy"],
                "log_likelihood": d["log_likelihood"],
            })
        hives.sort(key=lambda hv: hv["hive_id"])
        out[tup] = {"hives": hives}
    return out


def make_cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["load", "verify"])
    ap.add_argument("results_dir")
    args = ap.parse_args()
    if args.cmd == "load":
        r = load_results(args.results_dir)
        for key in r:
            print(f"{key}: {len(r[key]['hives'])} hives")
    else:
        n = verify_dir(args.results_dir)
        print(f"verified {n} files OK")


if __name__ == "__main__":
    make_cli()
