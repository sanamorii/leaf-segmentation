
import datetime
import os
import time
from pathlib import Path
import click
import logging

import numpy as np

from typing import Optional
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.amp import GradScaler
from torch.optim import Optimizer

from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

from leaf_seg.common.config import InstanceFinetuneConfig
from leaf_seg.dataset.plantdreamer_instance import build_dataloaders
from leaf_seg.instance.build import build_reporter, setup_maskrcnn
from leaf_seg.models.maskrcnn_torch import get_model as get_maskrcnn
from leaf_seg.models.utils import create_maskrcnn_ckpt, save_ckpt, load_ckpt
from leaf_seg.reporter.instance import InstanceTrainingReporter
from leaf_seg.common.verbose import get_tqdm_bar, resolve_progress_flag, suppress_stout
from leaf_seg.instance.train import fit

logger = logging.getLogger(__name__)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in ckpt:
                return ckpt[key]
    return ckpt


def load_pretrained_weights(model, ckpt_path, device, strict_load=False):
    ckpt = load_ckpt(ckpt_path, map_location=device)
    state_dict = _extract_state_dict(ckpt)

    if strict_load:
        model.load_state_dict(state_dict, strict=True)
        logger.info("Loaded pretrained weights (strict) from %s", ckpt_path)
        return

    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and value.shape == model_state[key].shape:
            filtered[key] = value
        else:
            skipped.append(key)

    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
    logger.info(
        "Loaded %d/%d tensors from %s (skipped %d mismatched)",
        len(filtered),
        len(state_dict),
        ckpt_path,
        len(skipped),
    )
    if missing_keys:
        logger.info("Missing keys after load: %s", missing_keys)
    if unexpected_keys:
        logger.info("Unexpected keys after load: %s", unexpected_keys)


def freeze_backbone(model: nn.Module):
    for param in model.backbone.parameters():
        param.requires_grad = False
    logger.info("Backbone frozen")


def unfreeze_backbone(model: nn.Module):
    for param in model.backbone.parameters():
        param.requires_grad = True
    logger.info("Backbone unfrozen")


def build_finetune_optimiser(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> Optimizer:
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})

    if not param_groups:
        raise RuntimeError("No trainable parameters found for finetuning.")

    return torch.optim.AdamW(param_groups, lr=head_lr, weight_decay=weight_decay)


def finetune(
    model: nn.Module,
    cfg: InstanceFinetuneConfig,
    train_loader,
    val_loader,
    reporter: InstanceTrainingReporter,
):

    backbone_lr = cfg.backbone_lr if cfg.backbone_lr is not None else cfg.lr * 0.1
    head_lr = cfg.head_lr if cfg.head_lr is not None else cfg.lr

    freeze_epochs = cfg.freeze_epochs
    freeze_backbone_flag = cfg.freeze_backbone
    remaining_epochs = cfg.epochs
    if freeze_epochs > 0:
        freeze_epochs = min(freeze_epochs, cfg.epochs)
        freeze_backbone_flag = True

        if freeze_backbone_flag:
            freeze_backbone(model)

        optimiser = build_finetune_optimiser(
            model=model,
            backbone_lr=backbone_lr,
            head_lr=head_lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="max", factor=0.5, patience=3
        )

        fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimiser=optimiser,
            scheduler=scheduler,
            cfg=cfg,
            reporter=reporter,
            start_epoch=0,
            end_epoch=freeze_epochs
        )
        remaining_epochs = cfg.epochs - freeze_epochs
        if remaining_epochs <= 0:
            return

        unfreeze_backbone(model)

    if freeze_backbone_flag and freeze_epochs == 0:
        freeze_backbone(model)

    optimiser = build_finetune_optimiser(
        model=model,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=3
    )

    start_epoch = cfg.epochs - remaining_epochs
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimiser=optimiser,
        scheduler=scheduler,
        cfg=cfg,
        reporter=reporter,
        start_epoch=start_epoch,
        end_epoch=cfg.epochs
    )


def run(cfg: InstanceFinetuneConfig):
    device = torch.device(cfg.device)

    train_loader, val_loader, spec = build_dataloaders(
        dataset_id=cfg.dataset,
        registry_path="data/datasets.yaml",
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = setup_maskrcnn(num_classes=cfg.num_classes, dataset=cfg.dataset, device=cfg.device)
    model.to(device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    cfg.output = os.path.join(cfg.output, f"{timestamp}-finetune-{model.name}")
    cfg.progress = resolve_progress_flag(cfg.progress)

    reporter = None
    if not cfg.no_report:
        reporter = build_reporter(cfg=cfg)

    load_pretrained_weights(model, cfg.ckpt, device=cfg.device, strict_load=cfg.strict_load)
    finetune(model, cfg, train_loader=train_loader, val_loader=val_loader, reporter=reporter)