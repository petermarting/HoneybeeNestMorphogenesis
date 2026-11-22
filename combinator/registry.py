"""Registry of the model checkpoints committed with this repository.

``V2_MODELS`` are the eight checkpoints of the paper's masked-convolutional
autoregressive model (the ``spatial_classifier_v2_*`` family,
``interaction=1``, ``embedding_dim=128``, ``kernel_size=3``): a k=1 global
baseline plus the 3 x 1D, 2D and 3D spatial-axis configurations at kernel 3.
Figure 6 averages the 1D/2D/3D configurations and shows the k=1 baseline
dashed.  All eight load with ``combinator.models.from_config`` at the canonical
grid geometry (20, 52, 104) -- see ``MANIFEST.md`` in ``models/`` for the
per-file configuration and provenance.
"""

from __future__ import annotations

# Each entry: (filename, key tuple (k, d, h, w, pos, ctx, int))
# Filenames are relative to the package's models directory.
V2_MODELS = [
    ("spatial_classifier_v2_k1_dFalse_hFalse_wFalse_posFalse_ctxFalse_int0.pth",
     (1, False, False, False, False, False, 0)),
    ("spatial_classifier_v2_k3_dFalse_hFalse_wTrue_posTrue_ctxTrue_int1.pth",
     (3, False, False, True, True, True, 1)),
    ("spatial_classifier_v2_k3_dFalse_hTrue_wFalse_posTrue_ctxTrue_int1.pth",
     (3, False, True, False, True, True, 1)),
    ("spatial_classifier_v2_k3_dTrue_hFalse_wFalse_posTrue_ctxTrue_int1.pth",
     (3, True, False, False, True, True, 1)),
    ("spatial_classifier_v2_k3_dTrue_hFalse_wTrue_posTrue_ctxTrue_int1.pth",
     (3, True, False, True, True, True, 1)),
    ("spatial_classifier_v2_k3_dTrue_hTrue_wFalse_posTrue_ctxTrue_int1.pth",
     (3, True, True, False, True, True, 1)),
    ("spatial_classifier_v2_k3_dFalse_hTrue_wTrue_posTrue_ctxTrue_int1.pth",
     (3, False, True, True, True, True, 1)),
    ("spatial_classifier_v2_k3_dTrue_hTrue_wTrue_posTrue_ctxTrue_int1.pth",
     (3, True, True, True, True, True, 1)),
]

def v2_config(key):
    """Turn a V2 key tuple into a SpatialClassifier config dict."""
    k, d, h, w, pos, ctx, inter = key
    return {
        "num_classes": 6, "embedding_dim": 128, "kernel_size": k, "dilation": 1,
        "depth": d, "height": h, "width": w, "pos_embed": pos,
        "context": ctx and k > 1, "interaction": inter,
    }
