"""Spatial alignment and cropping of nest grids.

Each colony is stored as an array of shape (weeks, 20, Hc, Wc) where the 20
"frame sides" alternate Side A / Side B of each of the 10 frames:

    index  0     1     2     3   ...  18    19
    side  A f1  B f1  A f2  B f2 ...  A f10 B f10

The depth axis (length 20) of the model therefore interleaves both sides of
every frame, and is "physical": frame order along this axis is the actual
order of frames in the nest.

Pipeline (identical to the original notebook):

1. ``map_labels``: remap the 21 detailed classes to 6 simplified classes.
2. ``align_grids``: phase-cross-correlation of each frame's binary wood mask
   to a global mean wood mask (Side A first, then mirrored Side B to the
   global mean, finally Side B to aligned Side A).  Integer-pixel shifts only.
3. ``crop_grids``: bounding box of the largest contour of the mean wood mask
   (``THRESH``) with ``MARGIN`` pixels of padding, cropped on height/width.
4. ``clean_invalid`` + ``relabel_consecutive``: dense class indices.

The result of the full pipeline is the training array used by the paper:
grids of shape (weeks, 20, 52, 104) for the 12 colonies (see the
``(D, H, W) = (20, 52, 104)`` note in ``models.py`` for why the height is 52
rather than the 67 quoted in some early drafts).
"""

from __future__ import annotations

import warnings

import numpy as np
from skimage import measure
from skimage.registration import phase_cross_correlation


def align_frame_skimage_with_shift(reference_frame, moving_frame, original_frame,
                                    grid_index=None, frame_indices=None):
    """Compute an alignment shift using binary frames (reference and moving)
    by phase cross-correlation, then apply the computed shift to the original
    (label) frame.  Returns a zero-frame if the shift cannot be computed."""
    if np.var(reference_frame) == 0 or np.var(moving_frame) == 0:
        if grid_index is not None and frame_indices is not None:
            print(f"Zero variance detected in grid {grid_index}, "
                  f"frame indices {frame_indices}")
        return np.zeros_like(original_frame)

    try:
        shift, error, diffphase = phase_cross_correlation(
            reference_frame, moving_frame, upsample_factor=1)
    except Exception as exc:  # noqa: BLE001 - fall back to no shift
        if grid_index is not None and frame_indices is not None:
            print(f"Error during cross-correlation in grid {grid_index}, "
                  f"frame indices {frame_indices}: {exc}")
        return np.zeros_like(original_frame)

    if np.isnan(error) or np.isinf(error):
        if grid_index is not None and frame_indices is not None:
            print(f"Invalid error value detected in grid {grid_index}, "
                  f"frame indices {frame_indices}")
        return np.zeros_like(original_frame)

    shift = np.round(shift).astype(int)
    x_shift, y_shift = shift

    aligned_frame = np.zeros_like(original_frame)
    if x_shift > 0:  # shift down
        aligned_frame[x_shift:, :] = original_frame[:-x_shift, :]
    elif x_shift < 0:  # shift up
        aligned_frame[:x_shift, :] = original_frame[-x_shift:, :]
    else:
        aligned_frame[:, :] = original_frame[:, :]

    if y_shift > 0:  # shift right
        aligned_frame[:, y_shift:] = aligned_frame[:, :-y_shift]
    elif y_shift < 0:  # shift left
        aligned_frame[:, :y_shift] = aligned_frame[:, -y_shift:]

    return aligned_frame


def align_grids(hive_grids: list[np.ndarray]) -> list[np.ndarray]:
    """Align all frames of all colonies to a global mean wood mask and to
    each other (Side B to aligned Side A)."""
    # Step 1: global mean wood class over all colonies (Side A only).
    side1_frames = []
    for hive_data in hive_grids:
        side1_grid = hive_data[:, ::2, :, :]  # Side A
        side1_frames.append(side1_grid)

    side1_frames_combined = np.concatenate(
        [g.reshape(-1, g.shape[-2], g.shape[-1]) for g in side1_frames], axis=0)
    mean_wood_class_1 = np.mean(
        (side1_frames_combined == 1).astype(np.float32), axis=0)  # global mean

    # Step 2: align both sides to the global mean, then Side B to Side A.
    hive_grids_aligned = []

    for grid_index, hive_data in enumerate(hive_grids):
        side1_grid = hive_data[:, ::2, :, :]      # Side A
        side2_grid = hive_data[:, 1::2, :, :]     # Side B
        mirrored_side2_grid = np.flip(side2_grid, axis=-1)

        aligned_side1_grid = np.zeros_like(side1_grid)
        aligned_side2_grid = np.zeros_like(mirrored_side2_grid)

        for i in range(side1_grid.shape[0]):      # weeks
            for j in range(side1_grid.shape[1]):  # frame sides
                aligned_side1_grid[i, j] = align_frame_skimage_with_shift(
                    mean_wood_class_1,
                    (side1_grid[i, j] == 1).astype(np.float32),
                    side1_grid[i, j],
                    grid_index=grid_index, frame_indices=(i, j))

                aligned_side2_grid[i, j] = align_frame_skimage_with_shift(
                    mean_wood_class_1,
                    (mirrored_side2_grid[i, j] == 1).astype(np.float32),
                    mirrored_side2_grid[i, j],
                    grid_index=grid_index, frame_indices=(i, j))

        final_aligned_side2_grid = np.zeros_like(aligned_side2_grid)
        for i in range(aligned_side1_grid.shape[0]):
            for j in range(aligned_side1_grid.shape[1]):
                final_aligned_side2_grid[i, j] = align_frame_skimage_with_shift(
                    (aligned_side1_grid[i, j] >= 1).astype(np.float32),
                    (aligned_side2_grid[i, j] >= 1).astype(np.float32),
                    aligned_side2_grid[i, j],
                    grid_index=grid_index, frame_indices=(i, j))

        combined_grid = np.zeros_like(hive_data)
        combined_grid[:, ::2, :, :] = aligned_side1_grid
        combined_grid[:, 1::2, :, :] = final_aligned_side2_grid
        hive_grids_aligned.append(combined_grid)

    return hive_grids_aligned


def poly_area(contour: np.ndarray) -> float:
    """Shoelace area of a (y, x) polyline."""
    y, x = contour[:, 0], contour[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compute_crop_bbox(hive_grids_aligned: list[np.ndarray],
                      margin: int = 1, thresh: float = 0.5):
    """Bounding box (inclusive) of the largest contour of the mean wood mask
    over all aligned grids, padded by ``margin`` and clipped to the array.

    Returns (rmin, rmax, cmin, cmax)."""
    h, w = np.asarray(hive_grids_aligned[0]).shape[-2:]
    all_frames = np.concatenate(
        [np.asarray(grid).reshape(-1, h, w) for grid in hive_grids_aligned],
        axis=0)

    wood_mask = (all_frames == 1).astype(np.float32)   # (F, H, W)
    mean_mask = wood_mask.mean(axis=0)                 # (H, W) in [0, 1]

    contours = measure.find_contours(mean_mask, level=thresh)
    if not contours:
        raise RuntimeError(f"No contour found on mean mask at threshold "
                           f"{thresh} - adjust THRESH.")
    largest = max(contours, key=poly_area)

    rmin, rmax = int(np.floor(largest[:, 0].min())), int(np.ceil(largest[:, 0].max()))
    cmin, cmax = int(np.floor(largest[:, 1].min())), int(np.ceil(largest[:, 1].max()))

    rmin = max(0, rmin - margin)
    rmax = min(h - 1, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(w - 1, cmax + margin)

    return (rmin, rmax, cmin, cmax)


def crop_last2(a: np.ndarray, bbox) -> np.ndarray:
    """Crop the last two dimensions by an inclusive (rmin, rmax, cmin, cmax)."""
    rmin, rmax, cmin, cmax = bbox
    a = np.asarray(a)
    sl = (slice(None),) * (a.ndim - 2) + (slice(rmin, rmax + 1), slice(cmin, cmax + 1))
    return a[sl]


def crop_grids(hive_grids_aligned: list[np.ndarray], margin: int = 1,
               thresh: float = 0.5) -> list[np.ndarray]:
    """Crop every aligned grid to the shared wood bbox."""
    bbox = compute_crop_bbox(hive_grids_aligned, margin=margin, thresh=thresh)
    return [crop_last2(g, bbox) for g in hive_grids_aligned]


def process_grids(hive_grids: list[np.ndarray],
                  crop: bool = True,
                  clean: bool = True,
                  margin: int = 1,
                  thresh: float = 0.5) -> list[np.ndarray]:
    """Full spatial preprocessing pipeline.

    Args:
        hive_grids: per-colony 6-class grids, shape (weeks, 20, Hc, Wc).
        crop: apply the mean-wood bounding-box crop (height/width).
        clean: zero out any leftover INVALID_LABEL (255) voxels and
            renumber remaining classes to consecutive indices.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        aligned = align_grids(hive_grids)
    if crop:
        grids = crop_grids(aligned, margin=margin, thresh=thresh)
    else:
        grids = aligned
    if clean:
        grids = clean_and_relabel(grids)
    return grids


def clean_and_relabel(grids: list[np.ndarray]) -> list[np.ndarray]:
    """Zero leftover invalid voxels and make class indices consecutive."""
    from .labels import clean_invalid
    from .labels import relabel_consecutive

    clean_invalid(grids)
    return relabel_consecutive(grids)
