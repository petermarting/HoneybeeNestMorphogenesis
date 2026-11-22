"""Training routine (PyTorch Lightning) for the spatial classifier.

Reproduces the paper's optimizer and schedule: AdamW (base lr 1e-3, weight
decay 1e-5, betas (0.9, 0.95)), class-embedding and positional-bias groups
trained at 10x the base lr, a 5-epoch linear warmup followed by cosine
annealing, batch size 8, 200 epochs, and a class-weighted cross-entropy loss
(weights = inverse training-set class frequencies).

Training augmentation follows the paper (models/MANIFEST.md): random width-axis
flips with p = 0.5, enabled via ``flip_aug=True``.  Augmentation is training
only -- prediction/evaluation never augments (see ``combinator.eval``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from .dataset import GridDataset, compute_class_weights
from .models import SpatialClassifier


class FlipAugDataset(Dataset):
    """Wrap a ``GridDataset`` and flip each sampled grid along its width axis
    with p = 0.5 (the paper's training-only augmentation).  Input labels and
    targets are the same grid, so the flip is consistent."""

    def __init__(self, base: GridDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        grid, hive_idx, week_idx = self.base[idx]
        if bool(torch.rand(()) < 0.5):
            grid = grid[..., ::-1].contiguous()
        return grid, hive_idx, week_idx


class SpatialClassifierModule(pl.LightningModule):
    """Thin Lightning wrapper around a ``SpatialClassifier``."""

    def __init__(self, model, criterion, learning_rate, weight_decay,
                 warmup_steps, cosine_steps):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.cosine_steps = cosine_steps

    def forward(self, grids):
        return self.model(grids)

    def training_step(self, batch, batch_idx):
        grids, _, _ = batch
        grids = grids.to(self.device, dtype=torch.long)
        logits = self(grids)
        loss = self.criterion(logits, grids)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        grids, _, _ = batch
        grids = grids.to(self.device, dtype=torch.long)
        logits = self(grids)
        loss = self.criterion(logits, grids)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        param_groups = [
            {"params": self.model.embedding_weights,
             "lr": self.learning_rate * 10},  # 10x base lr for the embedding
        ]
        if hasattr(self.model, "positional_embedding"):
            param_groups.append({"params": [self.model.positional_embedding],
                                 "lr": self.learning_rate * 10})
        param_groups.append({
            "params": [p for n, p in self.model.named_parameters()
                       if "embedding" not in n],
            "lr": self.learning_rate,
        })

        optimizer = torch.optim.AdamW(param_groups,
                                      weight_decay=self.weight_decay,
                                      betas=(0.9, 0.95))
        # A zero-warmup run skips the ramp (LinearLR needs start_factor > 0).
        start_factor = 1 / self.warmup_steps if self.warmup_steps > 0 else 1.0
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=start_factor,
                        total_iters=max(1, self.warmup_steps)),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.cosine_steps),
                ],
                milestones=[self.warmup_steps]),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]


def train_spatial_classifier(hive_grids, seed: int = 42, **kwargs) -> SpatialClassifier:
    """Train one model over a list of per-colony (weeks, D, H, W) arrays.

    Defaults match the paper.  ``class_weights`` is computed here from the
    provided grids (inverse frequency) instead of relying on a global, as in
    the original notebook.
    """
    pl.seed_everything(seed, workers=True)

    batch_size = kwargs.get('batch_size', 8)
    embedding_dim = kwargs.get('embedding_dim', 128)
    kernel_size = kwargs.get('kernel_size', 3)
    dilation = kwargs.get('dilation', 1)
    max_epochs = kwargs.get('max_epochs', 200)
    learning_rate = kwargs.get('learning_rate', 1e-3)
    weight_decay = kwargs.get('weight_decay', 1e-5)
    warmup_epochs = kwargs.get('warmup_epochs', 5)
    device = kwargs.get('device', 'cpu')
    enable_progress_bar = kwargs.get('enable_progress_bar', True)

    # The 5-epoch warmup is the paper default; keep the schedule valid when
    # training for few epochs (e.g. a smoke test) by shrinking the warmup.
    if max_epochs <= warmup_epochs:
        warmup_epochs = max(0, (max_epochs - 1) // 2)
    cosine_epochs = max_epochs - warmup_epochs
    assert cosine_epochs >= 1, "max_epochs must exceed warmup_epochs"

    flip_aug = kwargs.get('flip_aug', False)

    dataset = GridDataset(hive_grids)

    # Class weighting: inverse training-set frequencies (computed, not global).
    class_weights = compute_class_weights(dataset).to(device)

    if flip_aug:
        dataset = FlipAugDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True)

    model = SpatialClassifier(
        num_classes=dataset.num_classes,
        embedding_dim=embedding_dim,
        kernel_size=kernel_size,
        dilation=dilation,
        depth=kwargs.get('depth', True),
        height=kwargs.get('height', True),
        width=kwargs.get('width', True),
        pos_embed=kwargs.get('pos_embed', False),
        context=kwargs.get('context', True),
        interaction=kwargs.get('interaction', 1),
        num_depth=kwargs.get('num_depth', dataset.shape[-3]),
        num_height=kwargs.get('num_height', dataset.shape[-2]),
        num_width=kwargs.get('num_width', dataset.shape[-1]),
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    total_batches = len(dataloader)
    warmup_steps = warmup_epochs * total_batches
    cosine_steps = cosine_epochs * total_batches

    spatial_module = SpatialClassifierModule(
        model=model, criterion=criterion,
        learning_rate=learning_rate, weight_decay=weight_decay,
        warmup_steps=warmup_steps, cosine_steps=cosine_steps,
    )

    if device.startswith("cuda") or (device == "cuda"):
        device_index = int(device.split(":")[1]) if ":" in device else 0
        accelerator = "gpu"
        devices = [device_index]
    elif device == "cpu":
        accelerator = "cpu"
        devices = None
    else:
        raise ValueError(f"Unsupported device string: {device}")

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=1,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(spatial_module, train_dataloaders=dataloader)

    # Lightning may leave the module on CPU after fit; honour the device the
    # caller asked for so the returned model is ready to use.
    return model.to(device)

def train_all_v2(hive_grids, models_dir: str = "models", seed: int = 42,
                 device: str = "cpu", **kwargs) -> None:
    """Reproduce the eight committed checkpoints from scratch.

    Trains one model per configuration in ``combinator.registry.V2_MODELS``
    (the k=1 global baseline plus the seven k=3 spatial-axis configurations)
    with the full paper recipe -- 200 epochs on the whole corpus, seed 42,
    training-only width-flip augmentation (see models/MANIFEST.md) -- and
    writes each ``state_dict`` to ``models_dir`` under the committed filename.

    This is the full reproduction entry point (many GPU-hours in total); the
    ``autoregressive_model`` notebook demonstrates the API with a short run
    and uses the committed checkpoints as-is.
    """
    import os

    from .registry import V2_MODELS, v2_config

    os.makedirs(models_dir, exist_ok=True)
    for fname, key in V2_MODELS:
        cfg = v2_config(key)
        model = train_spatial_classifier(
            hive_grids, seed=seed, device=device, flip_aug=True, **cfg)
        path = os.path.join(models_dir, fname)
        torch.save(model.state_dict(), path)
        print(f"trained {fname}")
