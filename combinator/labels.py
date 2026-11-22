"""Class taxonomy and label remapping for honeybee nest content masks.

The raw masks produced by the comb-image segmentation pipeline use 21
detailed content classes. The autoregressive model works with a simplified
6-class scheme:

    0 background, 1 wood, 2 empty comb, 3 brood, 4 pollen, 5 nectar/honey

Colonies are identified by name; ``DD`` = unshuffled control colonies and
``SH`` = physically (weekly) shuffled treatment colonies. The order of the
colony array in ``AllColoniesCombinedDS48v2.npy`` matches ``COLONY_NAMES``.
"""

from __future__ import annotations

import numpy as np

# Original (detailed) class names, indices 0-20 (last two unused/unknown).
COMBINED_NAMES = [
    'background', 'wood', 'empty_comb', 'eggs', 'larvae', 'pupae', 'pollen',
    'nectar', 'honey', 'queen_cup', 'queen_cell', 'bees', 'drone_empty_comb',
    'drone_eggs', 'drone_larvae', 'drone_pupae', 'drone_pollen', 'drone_nectar',
    'drone_honey', '?', '?',
]

# Simplified class names used by the model.
SIMPLE_NAMES = [
    'background',      # 0
    'wood',            # 1
    'empty_comb',      # 2
    'any_brood',       # 3
    'pollen',          # 4
    'nectar_honey',    # 5
]

# Map from detailed (combined) index to simplified index.
LABEL_MAP = {
    0: 0,   # background -> background
    1: 1,   # wood -> wood
    2: 2,   # empty_comb -> empty_comb
    3: 3,   # eggs -> any_brood
    4: 3,   # larvae -> any_brood
    5: 3,   # pupae -> any_brood
    6: 4,   # pollen -> pollen
    7: 5,   # nectar -> nectar_honey
    8: 5,   # honey -> nectar_honey
    9: 2,   # queen_cup -> empty_comb (empty cell)
    10: 3,  # queen_cell -> any_brood (contains developing queen)
    11: 0,  # bees -> background
    12: 2,  # drone_empty_comb -> empty_comb
    13: 3,  # drone_eggs -> any_brood
    14: 3,  # drone_larvae -> any_brood
    15: 3,  # drone_pupae -> any_brood
    16: 4,  # drone_pollen -> pollen
    17: 5,  # drone_nectar -> nectar_honey
    18: 5,  # drone_honey -> nectar_honey
    19: 0,  # ? -> background
    20: 0,  # ? -> background
}

DEFAULT_LABEL = 0  # fallback: background

# Colony order in the combined array (first six = controls, last six =
# physically shuffled treatment).
COLONY_NAMES = ['DD1', 'DD2', 'DD4', 'DD5', 'DD6', 'DD7',
                'SH2', 'SH3', 'SH4', 'SH5', 'SH6', 'SH7']

# Value marking "invalid / empty" cells in raw masks (kept for compatibility).
INVALID_LABEL = 255


def map_labels(grid: np.ndarray) -> np.ndarray:
    """Remap a single detailed 21-class grid to the 6-class scheme."""
    return np.vectorize(lambda x: LABEL_MAP.get(x, DEFAULT_LABEL))(grid)


def map_all_labels(hive_grids: list[np.ndarray]) -> list[np.ndarray]:
    """Remap a list of per-colony detailed grids to the 6-class scheme."""
    return [map_labels(hive) for hive in hive_grids]


def clean_invalid(grids: list[np.ndarray]) -> list[np.ndarray]:
    """Replace leftover INVALID_LABEL (255) voxels with background (0)."""
    for grid in grids:
        grid[grid == INVALID_LABEL] = 0
    return grids


def relabel_consecutive(grids: list[np.ndarray]) -> list[np.ndarray]:
    """Renumber the labels present across all grids to consecutive integers
    (ascending order of the original value) so the class indices are dense."""
    unique_labels = np.unique(np.concatenate(grids))
    label_mapping = {old: new for new, old in enumerate(sorted(unique_labels))}
    return [np.vectorize(label_mapping.get)(grid) for grid in grids]
