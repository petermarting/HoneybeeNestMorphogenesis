"""Names, colours and geometry constants for the comb-content analysis.

Taken verbatim from the original ``CombContent_Analysis.ipynb``
(import cell); colours are the paper's Figure 1--5 palette.
"""

from __future__ import annotations

#: Colonies in mask/table row order: 6 unshuffled (DD) followed by 6
#: physically shuffled (SH).  Colony DD3 is absent (swarmed).
colony_names = ['DD1', 'DD2', 'DD4', 'DD5', 'DD6', 'DD7',
                'SH2', 'SH3', 'SH4', 'SH5', 'SH6', 'SH7']

# 12 detailed content classes (Contents/CombMasks_Contents_*.npy)
content_names = [
    'background',
    'wood',
    'empty_comb',
    'eggs',
    'larvae',
    'pupae',
    'pollen',
    'nectar',
    'honey',
    'queen_cup',
    'queen_cell',
    'bees',
]
#: The nine classes present inside the frame (drops queen/bee classes).
short_names = content_names[:9]

# 5-class simplified content array (CombMasks_Contents_simple.npy)
simple_names = [
    'background',
    'empty_comb',
    'any_brood',
    'pollen',
    'nectar_honey',
]

# comb-type array (CombMasks_CombTypes_aligned.npy)
combtype_names = [
    'background',
    'wood',
    'worker',
    'drone',
    'queen',
    'bee',
]

# combined content x comb-type array (CombMasks_Combined_aligned.npy)
combined_names = [
    'background',
    'wood',
    'worker_empty_comb',
    'worker_eggs',
    'worker_larvae',
    'worker_pupae',
    'worker_pollen',
    'worker_nectar',
    'worker_honey',
    'queen_cup',
    'queen_cell',
    'bees',
    'drone_empty_comb',
    'drone_eggs',
    'drone_larvae',
    'drone_pupae',
    'drone_pollen',
    'drone_nectar',
    'drone_honey',
]

# Figure palette (matches the paper's Figures 1-5).
contentcolors = [
    (0 / 255, 0 / 255, 0 / 255),      # background
    (166 / 255, 32 / 255, 64 / 255),  # wood
    (64 / 255, 64 / 255, 128 / 255),  # empty_comb
    (96 / 255, 96 / 255, 191 / 255),  # eggs
    (128 / 255, 128 / 255, 255 / 255),  # larvae
    (159 / 255, 159 / 255, 119 / 255),  # pupae
    (191 / 255, 191 / 255, 14 / 255),  # pollen
    (223 / 255, 223 / 255, 135 / 255),  # nectar
    (255 / 255, 255 / 255, 255 / 255),  # honey
]
contenthex = [
    '#000000',   # background
    '#a62040',   # wood
    '#404080',   # empty_comb
    '#6060bf',   # eggs
    '#8080ff',   # larvae
    '#9f9f77',   # pupae
    '#bfbf0e',   # pollen
    '#dfdf87',   # nectar
    '#ffffff',   # honey
]
simplehex = [
    '#1f998a',   # any_brood
    '#67cc5c',   # pollen
    '#34618d',   # nectar_honey
]
treatmenthex = [
    '#31688e',   # DD (unshuffled control)
    '#35b779',   # SH (physically shuffled)
]

# Frame-interior crop margins (applied with slice notation to the last two
# dimensions of a comb mask; see ``combinator.content.util.crop_last2dims``).
xmincrop = 6
xmaxcrop = -6
ymincrop = 6
ymaxcrop = -4

#: ``true_week`` values whose photographs show combs after the colony died.
#: Filter out with ``df[~df.true_week.isin(colony_deadout_week)]`` (paper:
#: dead-colony weeks are excluded from longitudinal summaries).
colony_deadout_week = [23, 17, 25, 31, 32, 33, 35, 37, 40]

#: Physical width of a frame interior (cm); the pixel-per-centimetre factor is
#: derived from the cropped mask width (``combinator.content.data.px_to_cm``).
within_frame_width_cm = 43
