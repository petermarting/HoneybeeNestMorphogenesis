"""Dataset and class-weight helpers.

``GridDataset`` exposes the (colony, week) time points of a list of per-colony
arrays of shape (weeks, D, H, W) as ``(sample, hive_idx, week_idx)`` triples,
where D is the 20-frame depth axis.  It converts to a tensor lazily so the full
18-hive corpus is not held in memory as tensors.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class GridDataset(Dataset):
    """Dataset over a list of per-colony arrays, each (weeks, D, H, W).

    One sample = one colony-week grid of shape (D, H, W)."""

    def __init__(self, grids: list[np.ndarray]) -> None:
        self.grids = [torch.as_tensor(grid) for grid in grids]
        # Precompute the global index -> (colony, week) mapping.
        self.grid_mapping = []
        for grid_idx, grid in enumerate(self.grids):
            self.grid_mapping.extend(
                (grid_idx, local_idx) for local_idx in range(grid.shape[0]))

        self.num_classes = int(max(g.max().item() for g in self.grids)) + 1
        self.shape = (len(self), *self.grids[0].shape[1:])

    def __len__(self) -> int:
        return len(self.grid_mapping)

    def __getitem__(self, idx):
        hive_idx, week_idx = self.grid_mapping[idx]
        return self.grids[hive_idx][week_idx], hive_idx, week_idx


def compute_class_weights(dataset: GridDataset, normalize: bool = True) -> torch.Tensor:
    """Inverse-frequency per-class weights (optionally normalized to sum to 1).

    Used to counteract the class imbalance in the cross-entropy loss
    (see paper: "each ground-truth class's contribution to the loss was scaled
    by the inverse of its frequency")."""
    all_classes = torch.cat([g.flatten() for g in dataset.grids], dim=0)
    class_counts = torch.bincount(all_classes, minlength=dataset.num_classes)
    total = all_classes.numel()
    weights = total / (class_counts + 1e-6)
    if normalize:
        weights = weights / weights.sum()
    return weights


def load_grids(path: str) -> list[np.ndarray]:
    """Load a deposited/combined ``.npy`` (or a ``.zip`` containing a single
    ``.npy``; the 156 MB raw file is too large for GitHub's 100 MB/file cap,
    so it is committed zipped under ``data/``) of a (ragged) per-colony
    stack and return it as a Python list of arrays."""
    if str(path).endswith(".zip"):
        import io
        import zipfile

        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".npy")]
            if len(names) != 1:
                raise ValueError(f"expected a single .npy in {path!r}, got {names}")
            arr = np.load(io.BytesIO(zf.read(names[0])), allow_pickle=True)
    else:
        arr = np.load(path, allow_pickle=True)
    return [arr[i] for i in range(len(arr))]
