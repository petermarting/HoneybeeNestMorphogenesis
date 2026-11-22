"""Evaluation and simulated shuffling.

``evaluate_spatial_classifier`` runs a trained model over a list of per-colony
arrays (18 hives: 6 control, 6 physically shuffled, 6 manually shuffled) and
returns, per hive and per week, the ground truth, predicted class, Shannon
entropy and log-likelihood of the predicted distribution -- the quantities the
paper figures are built from.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import GridDataset


def shuffle_frames_in_pairs_np(hive_array: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Manually shuffle frames in (Side A, Side B) pairs along the depth axis.

    This is the "simulated shuffling" of the paper: it preserves the
    within-frame cross-side relationship (each pair stays adjacent) while
    destroying the global depth order.

    Args:
        hive_array: (time, depth, height, width), depth even.
        seed: optional RNG seed for reproducibility.
    """
    time, depth, height, width = hive_array.shape
    assert depth % 2 == 0, "Depth must be even for pair-wise shuffling."
    rng = np.random.default_rng(seed)
    frame_pairs = [(i, i + 1) for i in range(0, depth, 2)]
    shuffled = frame_pairs[:]
    rng.shuffle(shuffled)
    shuffled_depth_slices = [hive_array[:, [i, j], :, :] for i, j in shuffled]
    return np.concatenate(shuffled_depth_slices, axis=1)


def evaluate_spatial_classifier(model, hive_grids, device, batch_size=16):
    """Evaluate ``model`` on ``hive_grids``; returns a per-hive results dict.

    Args:
        model: a (nn.Module) SpatialClassifier (or wrapper) producing
            (B, C, D, H, W) logits.
        hive_grids: list of per-colony arrays (time, D, H, W).
        device: torch device.
    """
    dataset = GridDataset(hive_grids)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    hive_results = defaultdict(lambda: {
        'ground_truth': [], 'predictions': [],
        'entropy': [], 'log_likelihood': [],
    })

    with torch.no_grad():
        for grids, hive_idx, week_idx in eval_loader:
            grids = grids.to(device, dtype=torch.long)
            logits = model(grids)  # (B, C, D, H, W)

            if logits.dim() == 5:
                logits = logits.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
                B, D, H, W, C = logits.shape
            else:
                raise ValueError(
                    f"Expected 5D logits [B,C,D,H,W], got {logits.shape}")

            logits_flat = logits.reshape(-1, C)
            targets_flat = grids.reshape(-1)

            preds = torch.argmax(logits_flat, dim=1)
            probs = F.softmax(logits_flat, dim=1)
            log_probs = F.log_softmax(logits_flat, dim=1)
            entropy = -(probs * log_probs).sum(dim=1)
            log_likelihood = -F.cross_entropy(logits_flat, targets_flat,
                                              reduction='none')

            shape = (B, D, H, W)
            preds = preds.view(shape).cpu()
            entropy = entropy.view(shape).cpu()
            log_likelihood = log_likelihood.view(shape).cpu()
            targets = grids.cpu()

            for i in range(B):
                hid = hive_idx[i].item()
                hive_results[hid]['ground_truth'].append(targets[i])
                hive_results[hid]['predictions'].append(preds[i])
                hive_results[hid]['entropy'].append(entropy[i])
                hive_results[hid]['log_likelihood'].append(log_likelihood[i])

    results = {'hives': []}
    for hive_id, data in hive_results.items():
        results['hives'].append({
            'hive_id': hive_id,
            'ground_truth': torch.stack(data['ground_truth'], dim=0).numpy(),
            'predictions': torch.stack(data['predictions'], dim=0).numpy(),
            'entropy': torch.stack(data['entropy'], dim=0).numpy(),
            'log_likelihood': torch.stack(data['log_likelihood'], dim=0).numpy(),
        })
    # Keep the hive order deterministic (by hive_id).
    results['hives'].sort(key=lambda h: h['hive_id'])
    return results


def build_eval_hives(hive_grids, num_control: int = 6, seed: int = 42):
    """Build the 18-hive evaluation list:
        [0:6]   control (unshuffled)
        [6:12]  physically shuffled (treatment)
        [12:18] manually (simulated) shuffled copies of the controls.
    """
    control = hive_grids[:num_control]
    physical = hive_grids[num_control:2 * num_control]
    manual = [shuffle_frames_in_pairs_np(h, seed=seed + i)
              for i, h in enumerate(control)]
    return control + physical + manual
