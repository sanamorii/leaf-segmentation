from dataclasses import dataclass
import time
import logging
import datetime
import click
import sys
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Literal, Optional
from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as smp_losses
from segmentation_models_pytorch.base.model import SegmentationModel

from dataset.utils import rgb_to_class
from dataset.plantdreamer_semantic import get_dataloader
from loss.cedice import CEDiceLoss
from loss.earlystop import EarlyStopping
from models.utils import create_ckpt, save_ckpt, load_ckpt
from models.modelling import get_smp_model
from metrics import StreamSegMetrics

from leaf_seg.reporter.semantic import SemanticTrainingReporter
from leaf_seg.common.verbose import get_tqdm_bar, resolve_progress_flag
from leaf_seg.semantic.build import build_optimiser, build_reporter, build_scheduler, setup_model
from leaf_seg.semantic.config import SemanticTrainConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MetricLiterals = Literal[
    "overall_acc", 
    "mean_acc", 
    "fwavcc", 
    "mean_iou", 
    "mean_dice", 
    "class_iou", 
    "class_dice"
    ]


def validate_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    metrics: StreamSegMetrics,
    loss_fn,
    device: str,
    # epochs: tuple[int, int],
    # verbose: bool = True,
):
    start = time.time()
    
    model.eval()
    metrics.reset()

    running = {
        "loss_total": 0.0,
        "elapsed_time": 0.0,
    }
    batch_count = 0
    
    with torch.no_grad():
        for img, mask in loader:
            batch_count += 1
            img = img.to(device).float()
            mask = mask.to(device).long()

            output = model(img)
            # detaches tensor from comp graph, selects the highest score for each pixel
            # cpu() moves from gpu to cpu, numpy() converts from tensor to np array
            preds = output.detach().max(dim=1)[1].cpu().numpy()
            targets = mask.detach().cpu().numpy()
            
            metrics.update(targets, preds)

            # update metrics and validation loss
            loss = loss_fn(output, mask)
            loss_item = float(loss.item())
            running["loss_total"] += loss_item
            
            if hasattr(loader, "set_postfix"): 
                postfix = {'loss': float(loss.item())}
                if metrics is not None:
                    r = metrics.get_results()
                    postfix.update({
                        'mIoU': float(r["mean_iou"]),
                        'mDice': float(r["mean_dice"]),
                        'mAcc': float(r["mean_acc"]),
                    })
                loader.set_postfix(postfix)
    
    running["elapsed_time"] = time.time() - start
    running["loss_total"] = (running["loss_total"] / batch_count) if batch_count > 0 else 0.0
    
    if metrics is not None:
        r = metrics.get_results()
        running.update(r)
        # running.update({
        #     "overall_acc": float(r["Overall Acc"]),
        #     "mean_acc": float(r["Mean Acc"]),
        #     "fwavcc": float(r["FreqW Acc"]),
        #     "mean_iou": float(r["Mean IoU"]),
        #     "mean_dice": float(r["Mean Dice"]),
        #     "class_iou": r["Class IoU"],    # dict[int->float]
        #     "class_dice": r["Class Dice"]   # dict[int->float]
        # })

    return running


def train_epoch(
    model: nn.Module,
    loss_fn,
    optimiser: Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler], # mixed precision vs default precision (FP16)
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    # epochs: tuple[int, int],     
    clip_grad_norm: Optional[float] = None,  # prevent exploding gradients
    metrics: StreamSegMetrics = None,
    # verbose : bool = True,
) -> Dict[str, float | int]:
    start = time.time()

    model.train()
    batch_count = 0
    if metrics is not None: metrics.reset()

    running = {
        "loss_total": 0.0,
        "elapsed_time": 0.0,
    }

    for imgs, masks in loader:
        batch_count += 1
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # print("Target min/max:", masks.min().item(), masks.max().item())

        optimiser.zero_grad(set_to_none=True)
        if scaler is not None:
            device_type = 'cuda' if getattr(device, 'type', str(device)).startswith('cuda') else 'cpu'
            with torch.amp.autocast(device_type=device_type):
                outputs = model(imgs)
                loss = loss_fn(outputs, masks)
            
            scaler.scale(loss).backward()
            
            if clip_grad_norm is not None:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            scaler.step(optimiser)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            loss.backward()

            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            optimiser.step()

        if metrics is not None:
            with torch.no_grad():
                preds = outputs.argmax(dim=1)  # [B,H,W]

                gt = masks.detach().cpu().numpy()
                pr = preds.detach().cpu().numpy()

            metrics.update(gt, pr)

        loss_item = float(loss.item())
        running["loss_total"] += loss_item

        if hasattr(loader, "set_postfix"): 
            postfix = {'loss': float(loss.item())}
            if metrics is not None:
                r = metrics.get_results()
                postfix.update({
                    'mIoU': float(r["mean_iou"]),
                    'mDice': float(r["mean_dice"]),
                    'mAcc': float(r["mean_acc"]),
                })
            loader.set_postfix(postfix)

    running["elapsed_time"] = time.time() - start
    running["loss_total"] = (running["loss_total"] / batch_count) if batch_count > 0 else 0.0

    if metrics is not None:
        r = metrics.get_results()
        running.update(r)

    return running



def fit(
    model : nn.Module,
    loss_fn,
    optimiser: Optimizer,
    scheduler: LRScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: SemanticTrainConfig,
    reporter: SemanticTrainingReporter | None = None,
    save: bool = True,
    # monitor_metric: MetricLiterals  = "mean_iou",
    # gradient_clipping: float = 0.1,
):
    device = torch.device(cfg.device) if not isinstance(cfg.device, torch.device) else cfg.device
    model.to(device)

    best_vloss = float("inf")
    best_metric = float(-"inf")
    best_epoch = 0

    start_epoch = 0
    cur_itrs = 0

    # normalize device
    if not isinstance(device, torch.device):
        device = torch.device(device)

    # use the proper GradScaler constructor
    grad_scaler = torch.amp.GradScaler(device=device) if cfg.use_amp else None 
    stopper = EarlyStopping(patience=cfg.patience, delta=0.001)  # early stopping policy

    train_metrics = StreamSegMetrics(cfg.num_classes)
    val_metrics = StreamSegMetrics(cfg.num_classes)

    # resume from checkpoint if provided
    if cfg.resume is not None:
        ckpt = load_ckpt(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimiser.load_state_dict(ckpt["optimizer_state"])
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            logger.warning("Could not load scheduler state from checkpoint")

        cur_itrs = ckpt.get("cur_itrs", 0)
        start_epoch = int(ckpt.get("epoch", 0))

        validation_stats = ckpt.get("validation_stats", None)
        if validation_stats is not None:
            best_vloss = validation_stats["loss_total"]   
            best_metric = validation_stats[cfg.monitor_metric]

        logger.info("Resumed training from %s (starting epoch=%d)", cfg.resume, start_epoch)


    for epoch in range(start_epoch, cfg.epochs):
        
        train_stats = train_epoch(
            model=model,
            loader=get_tqdm_bar(train_loader, epoch, cfg.epochs, "Train", cfg.progress),
            loss_fn=loss_fn,
            optimiser=optimiser,
            scaler=grad_scaler,
            device=device,
            metrics=train_metrics,
            clip_grad_norm=cfg.gradient_clipping,
        )

        val_stats = validate_epoch(
            model=model,
            loader=get_tqdm_bar(val_loader, epoch, cfg.epochs, "Train", cfg.progress),
            loss_fn=loss_fn,
            metrics=val_metrics,
            device=device,
        )

        scheduler.step(val_stats["loss_total"])  # epoch param is deprecated


        # save model
        if save:
            checkpoint = create_ckpt(
                cur_itrs=cur_itrs,
                model=model,
                optimiser=optimiser,
                scheduler=scheduler,
                train_stats=train_stats["loss_total"],
                val_stats=val_stats["loss_total"],
                epoch=epoch,
                num_classes=cfg.num_classes
            )

            # ensure checkpoints directory exists
            ckpt_dir = Path(cfg.out) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)


            # update best_vloss and best_score based on monitor_metric
            if val_stats["loss_total"] < best_vloss:
                best_vloss = float(val_stats["loss_total"])
            
            model_name = getattr(model, 'name', model.__class__.__name__)
            save_ckpt(checkpoint, str(ckpt_dir / f"{model_name}-{cfg.epochs}_current.pth"))
            metric_value = val_stats.get(cfg.monitor_metric, None)

            if metric_value is not None and metric_value > best_metric:
                best_metric = metric_value
                save_ckpt(checkpoint, str(ckpt_dir / f"{model_name}-{cfg.epochs}_best.pth"))
            

        logger.info(
            "Epoch %d/%d: Avg Train Loss: %.4f, Avg Val Loss: %.4f, Mean IoU: %.4f, Training time: %s, Validation time: %s",
            epoch+1, cfg.epochs,
            train_stats["loss_total"],
            val_stats["loss_total"],
            val_stats.get(cfg.monitor_metric, 0.0),
            str(datetime.timedelta(seconds=int(train_stats["elapsed_time"]))),
            str(datetime.timedelta(seconds=int(val_stats["elapsed_time"])))
        )


        if reporter is not None:
            lr_value = None
            if optimiser is not None and hasattr(optimiser, "param_groups") and optimiser.param_groups:
                lr_value = float(optimiser.param_groups[0].get("lr", 0.0))
            reporter.log_epoch(
                epoch=epoch + 1,
                epochs=cfg.epochs,
                train_stats=train_stats,
                val_stats=val_stats,
                lr=lr_value,
            )
            
        stopper(best_vloss)
        if stopper.early_stop:
            logger.info("No improvement - terminating.")
            break


        if device.type == "cuda": torch.cuda.empty_cache()  # clear cache

def run(cfg: SemanticTrainConfig):
    cfg.progress = resolve_progress_flag(cfg.progress)

    train_loader, val_loader = get_dataloader(
        dataset=cfg.dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        num_classes=cfg.num_classes,
    )

    model = setup_model(cfg)
    optimiser = build_optimiser(model, cfg.lr)
    scheduler = build_scheduler(optimiser)
    loss_fn = CEDiceLoss(ce_weight=0.5, dice_weight=0.5)

    reporter = None
    if not cfg.no_report:
        reporter = build_reporter(cfg, model_name = model.name)

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimiser=optimiser,
        scheduler=scheduler,
        loss_fn=loss_fn,
        cfg=cfg,
        reporter=reporter,
    )