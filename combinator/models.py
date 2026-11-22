"""The masked-convolutional autoregressive spatial classifier.

This is the model described in the paper's "Masked-convolutional
autoregressive model" methods section.  Its forward pass is:

    1. Class embedding lookup: each voxel's integer label -> ``embedding_dim``
       vector (a learned ``num_classes x embedding_dim`` table plus a bias).
    2. Center-masked 3-D convolution: a ``MaskedConv3d`` whose center voxel is
       zeroed so a voxel can never copy its own label.  The kernel size is
       ``k`` along each *included* spatial axis and 1 along excluded axes.
    3. Optional learnable positional bias (broadcast across excluded axes).
    4. Optional first-order (multilinear) interaction between context and
       position (residual, two pointwise convs + elementwise product).
    5. Tied-weight readout: dot of the output vector with the *same* class
       embedding table -> logits over classes.

Input:  long tensor of shape (batch, D, H, W) of class labels.
Output: floats of shape (batch, C, D, H, W) (logits; caller softmaxes).

Canonical grid geometry
-----------------------
The committed checkpoints were trained on grids of shape
(D, H, W) = (20, 52, 104): 20 comb sides (10 frames x 2 sides) along the
depth axis, cropped to 52 x 104 in height/width by the mean-wood bbox (see
``preprocessing``).  These are the values baked into the positional-embedding
weights; the model is geometry-specific, so they must match at load time.

(Note: an early draft of the methods text describes the pre-crop /
``ContentsSimple`` grid as 20 x 67 x 103.  The weights in this repository are
the ground truth and correspond to 20 x 52 x 104.)
"""

from __future__ import annotations

import math
import re
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


#: Grid geometry of the committed checkpoints: (depth, height, width) =
#: 20 comb sides x 52 x 104.
CANONICAL_GRID_SHAPE = (20, 52, 104)


def lecun_normal_masked_conv3d(tensor: torch.Tensor, mask_center: bool = True):
    """LeCun-normal init for the (center-masked) conv kernel; the masked
    center pixel is excluded from the fan-in so the variance stays correct."""
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The tensor to initialize must be a PyTorch Tensor.")
    fan_in = tensor.size(1) * tensor.size(2) * tensor.size(3) * tensor.size(4)
    if mask_center:
        fan_in -= 1  # exclude the masked center pixel
    std = math.sqrt(1.0 / fan_in)
    with torch.no_grad():
        tensor.normal_(0, std)


class MaskedConv3d(nn.Conv3d):
    """``nn.Conv3d`` whose kernel center is zero on every forward pass so that
    a voxel's own label can never contribute to its prediction."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)
        lecun_normal_masked_conv3d(self.weight, mask_center=True)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        self.mask_center()
        return super().forward(input)

    def mask_center(self):
        with torch.no_grad():
            center = tuple(dim // 2 for dim in self.weight.shape[2:])
            self.weight[:, :, center[0], center[1], center[2]] = 0.0


class MultilinearConv3d(nn.Module):
    """Efficient first-order interaction layer: one pointwise conv produces
    ``(order + 1)`` feature vectors that are multiplied together and added
    back to the input (residual).  ``order`` >= 1 keeps a single
    ``context * position``-style nonlinearity with bounded complexity."""

    def __init__(self, embedding_dim, order):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.order = max(1, order)
        self.num_chunks = self.order + 1

        self.conv = nn.Conv3d(embedding_dim, self.num_chunks * embedding_dim,
                              kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in',
                                nonlinearity='linear')

    def forward(self, x):
        # x: (B, embedding_dim, D, H, W)
        transformed = self.conv(x)  # (B, num_chunks*emb, D, H, W)
        bsz, _, d, h, w = transformed.size()
        transformed = transformed.view(bsz, self.num_chunks, self.embedding_dim,
                                       d, h, w)
        interaction = transformed[:, 0] * transformed[:, 1]
        out = interaction
        for idx in range(2, self.num_chunks):
            interaction = interaction * transformed[:, idx]
            out = out + interaction
        return x + out


class SpatialClassifier(nn.Module):
    """Autoregressive masked-convolutional classifier (see module docstring)."""

    def __init__(self, num_classes, num_depth, num_height, num_width,
                 embedding_dim=128, kernel_size=3, dilation=1,
                 depth=True, height=True, width=True, pos_embed=False,
                 context=True, interaction=1):
        super().__init__()

        kwargs = self.get_kwargs(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            depth=depth, height=height, width=width,
            pos_embed=pos_embed, context=context,
            num_depth=num_depth, num_height=num_height, num_width=num_width,
            interaction=interaction,
        )

        self.num_classes = kwargs['num_classes']
        self.embedding_dim = kwargs['embedding_dim']
        self.rsqrt_dim = self.embedding_dim ** -0.5
        self.context = kwargs['context']
        self.pos_embed = kwargs['pos_embed']
        self.num_depth = kwargs['num_depth']
        self.num_height = kwargs['num_height']
        self.num_width = kwargs['num_width']
        self.interaction = kwargs['interaction']

        # Tied input/output class-embedding table and a per-voxel bias.
        self.embedding_weights = nn.Parameter(
            torch.randn(self.num_classes, self.embedding_dim) * self.rsqrt_dim)
        self.bias = nn.Parameter(
            torch.randn(self.embedding_dim) * self.rsqrt_dim)

        if self.context:
            spatial_dims = (
                kwargs['kernel_size'] if kwargs['depth'] else 1,
                kwargs['kernel_size'] if kwargs['height'] else 1,
                kwargs['kernel_size'] if kwargs['width'] else 1,
            )
            self.conv = MaskedConv3d(
                self.embedding_dim, self.embedding_dim, spatial_dims,
                padding="same", bias=False, dilation=kwargs['dilation'])
            self.multilinear = (MultilinearConv3d(self.embedding_dim,
                                                  self.interaction)
                                if self.interaction >= 1 else None)
        else:
            self.conv = None
            self.multilinear = None

        if self.pos_embed:
            self.positional_embedding = nn.Parameter(
                torch.randn(1, self.embedding_dim, self.num_depth,
                            self.num_height, self.num_width) * self.rsqrt_dim)

    @staticmethod
    def get_kwargs(num_classes, embedding_dim, kernel_size, dilation,
                   depth, height, width, pos_embed, context,
                   num_depth, num_height, num_width, interaction):
        """Normalize and validate configuration flags (mirrors the original)."""
        if context and kernel_size <= 1:
            warnings.warn("`context` is True but `kernel_size` <= 1; "
                          "setting `context` to False.")
            context = False

        if not (pos_embed or context):
            depth = height = width = False

        if not context:
            kernel_size = 1

        if not (depth or height or width):
            if pos_embed:
                warnings.warn("`pos_embed` True but depth/height/width all "
                              "False; setting `pos_embed` to False.")
                pos_embed = False
            if context:
                warnings.warn("`context` True but depth/height/width all "
                              "False; setting `context` to False.")
                context = False

        interaction = _coerce_interaction(interaction, context)

        normalized_kernel = kernel_size if any([depth, height, width]) else 1

        return {
            'num_classes': num_classes,
            'embedding_dim': embedding_dim,
            'kernel_size': normalized_kernel,
            'dilation': dilation,
            'depth': depth,
            'height': height,
            'width': width,
            'pos_embed': pos_embed,
            'context': context,
            'num_depth': num_depth if depth else 1,
            'num_height': num_height if height else 1,
            'num_width': num_width if width else 1,
            'interaction': interaction,
        }

    def forward(self, x):
        """x: (B, D, H, W) long labels -> (B, C, D, H, W) logits."""
        bsz, d, h, w = x.size()
        embedding_weights = self.embedding_weights + self.bias

        if self.context:
            x = F.embedding(x, embedding_weights)
            x = rearrange(x, 'b d h w c -> b c d h w')
            x = self.conv(x) + self.bias.view(1, -1, 1, 1, 1)
            if self.pos_embed:
                x = x + self.positional_embedding
            if self.multilinear:
                x = self.multilinear(x)
            logits = torch.einsum('b c d h w, co -> b o d h w',
                                  x, embedding_weights.T)
        else:
            if self.pos_embed:
                x = self.positional_embedding + self.bias.view(1, -1, 1, 1, 1)
                logits = torch.einsum('b c d h w, co -> b o d h w',
                                      x, embedding_weights.T)
                logits = logits.expand(bsz, self.num_classes, d, h, w)
            else:
                logits = torch.einsum('c, co -> o', self.bias,
                                      embedding_weights.T)
                logits = logits.view(1, self.num_classes, 1, 1, 1).expand(
                    bsz, self.num_classes, d, h, w)

        return logits


def _coerce_interaction(interaction, context):
    if not isinstance(interaction, (int, float)) or interaction < 0:
        warnings.warn("`interaction` must be numeric >= 0; setting to 0.")
        interaction = 0
    elif interaction >= 1 and not context:
        warnings.warn("`interaction` >= 1 requires `context`; setting to 0.")
        interaction = 0
    return int(interaction)


def from_config(config: dict, num_depth=20, num_height=52, num_width=104,
                device='cpu') -> SpatialClassifier:
    """Build a model from a config dict (the normalized ``get_kwargs`` form).
    ``num_depth/height/width`` are the full spatial sizes (the grid geometry),
    matching the committed weights (20, 52, 104)."""
    return SpatialClassifier(
        num_classes=config['num_classes'],
        num_depth=num_depth, num_height=num_height, num_width=num_width,
        embedding_dim=config['embedding_dim'],
        kernel_size=config['kernel_size'],
        depth=config['depth'], height=config['height'], width=config['width'],
        pos_embed=config['pos_embed'], context=config['context'],
        interaction=config['interaction'],
    ).to(device)


def load(path, num_classes=6, num_depth=20, num_height=52, num_width=104,
         embedding_dim=128, kernel_size=3, depth=True, height=True, width=True,
         pos_embed=True, context=True, interaction=1, device='cpu') -> SpatialClassifier:
    """Load a stored state dict into a freshly-built ``SpatialClassifier``.

    The flags describe the architecture that produced the checkpoint; pass
    the values that match the filename.  ``weights_only=True`` is used
    (checkpoints are plain ``state_dict`` objects)."""
    import torch as _t
    model = SpatialClassifier(
        num_classes=num_classes,
        num_depth=num_depth, num_height=num_height, num_width=num_width,
        embedding_dim=embedding_dim, kernel_size=kernel_size,
        depth=depth, height=height, width=width,
        pos_embed=pos_embed, context=context, interaction=interaction,
    )
    state = _t.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device)


# ---- filename helpers --------------------------------------------------

# v2 (paper) checkpoint: spatial_classifier_v2_k<k>_d{b}_h{b}_w{b}_pos{b}_ctx{b}_int<n>
_V2_RE = re.compile(
    r"spatial_classifier_v2_k(?P<k>\d+)_d(?P<d>True|False)_h(?P<h>True|False)"
    r"_w(?P<w>True|False)_pos(?P<pos>True|False)_ctx(?P<ctx>True|False)"
    r"_int(?P<int>\d+)$")

def _stem(path: str) -> str:
    """Basename without its final extension (e.g. 'a/b.pth' -> 'b')."""
    name = path.split('/')[-1]
    base, dot, ext = name.rpartition('.')
    return base if dot and not ext.isdigit() else name


def config_from_v2_name(filename: str) -> dict:
    stem = _stem(filename)
    m = _V2_RE.search(stem)
    if not m:
        raise ValueError(f"{filename!r} does not look like a v2 checkpoint")
    k = int(m['k'])
    depth, height, width = m['d'] == 'True', m['h'] == 'True', m['w'] == 'True'
    ctx = m['ctx'] == 'True'
    pos = m['pos'] == 'True'
    context = ctx and k > 1
    return {
        'num_classes': 6, 'embedding_dim': 128,
        'kernel_size': k, 'dilation': 1,
        'depth': depth, 'height': height, 'width': width,
        'pos_embed': pos, 'context': context, 'interaction': int(m['int']),
    }


def load_any(path, num_depth=20, num_height=52, num_width=104, device='cpu'):
    """Load a v2 checkpoint, inferring its configuration from its filename.

    ``path`` must be a ``spatial_classifier_v2_*`` filename (see
    ``combinator/registry.py`` for the committed set).  The v1 ``global_k1``
    and ``embedding_k<k>`` ablation checkpoints used different architectures
    and are deliberately not supported by this loader.
    """
    import torch as _t

    stem = _stem(path)
    if not _V2_RE.search(stem):
        raise ValueError(
            f"Unrecognized checkpoint filename: {stem!r}. Expected a "
            f"spatial_classifier_v2_* pattern (see registry.V2_MODELS).")
    cfg = config_from_v2_name(stem)
    model = from_config(cfg, num_depth=num_depth, num_height=num_height,
                        num_width=num_width, device=device)
    state = _t.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model
