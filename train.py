
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
from segmentation.reporter.semantic import SemanticTrainingReporter
from segmentation.utils.verbose import get_tqdm_bar, resolve_progress_flag

MetricLiterals = Literal[
    "overall_acc", 
    "mean_acc", 
    "fwavcc", 
    "mean_iou", 
    "mean_dice", 
    "class_iou", 
    "class_dice"
    ]

@dataclass
class RunningStats:
    loss_total: float
    elapsed_time: float


# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def setup_model(model_name, encoder, dataset, num_classes, device):
    # create a simple model and optimizer
    model = get_smp_model(name=model_name, encoder=encoder, weights="imagenet", classes=num_classes)
    model.name = f"{model.__class__.__name__.lower()}-{encoder}-{dataset}"
    device = torch.device(device)
    return model.to(device)

def freeze_encoder(model):
    if hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False
        logger.info("Encoder frozen")
    else:
        logger.warning("Model has no encoder - freezing unsuccessful")

def unfreeze_encoder(model):
    if hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = True
        logger.info("Encoder unfrozen")
    else:
        logger.warning("Model has no encoder - unfreeze unsuccessful")

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
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered[k] = v
        else:
            skipped.append(k)

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


def build_finetune_optimiser(model, encoder_lr, decoder_lr, weight_decay):
    enc_params = []
    dec_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder" in name:
            enc_params.append(p)
        else:
            dec_params.append(p)

    param_groups = []
    if enc_params:
        param_groups.append({"params": enc_params, "lr": encoder_lr})
    if dec_params:
        param_groups.append({"params": dec_params, "lr": decoder_lr})

    if not param_groups:
        raise RuntimeError("No trainable parameters found for finetuning.")
    return torch.optim.AdamW(param_groups, lr=decoder_lr, weight_decay=weight_decay)


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
    epochs : int,
    device,
    num_classes : int,
    use_amp: bool = False,
    patience: int = 0,
    gradient_clipping: float = 0.1,
    # visualise: bool = False,
    monitor_metric: MetricLiterals  = "mean_iou",
    resume: str | None = None,
    start_epoch: int = 0,
    save: bool = True,
    progress: bool = True,
    reporter: SemanticTrainingReporter | None = None,
    directory: Path = Path('checkpoints')
):
    
    best_vloss = np.inf
    best_score = 0.0
    cur_itrs = 0

    # normalize device
    if not isinstance(device, torch.device):
        device = torch.device(device)

    # use the proper GradScaler constructor
    grad_scaler = torch.amp.GradScaler(device=device) if use_amp else None 
    loss_stop_policy = EarlyStopping(patience=patience, delta=0.001)  # early stopping policy


    train_metrics = StreamSegMetrics(num_classes)
    val_metrics = StreamSegMetrics(num_classes)

    # resume from checkpoint if provided
    if resume is not None:
        ckpt = load_ckpt(resume, map_location=device)
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
            best_score = validation_stats[monitor_metric]

        logger.info("Resumed training from %s (starting epoch=%d)", resume, start_epoch)


    for epoch in range(start_epoch, epochs):
        
        train_stats = train_epoch(
            model=model,
            loader=get_tqdm_bar(train_loader, epoch, epochs, "Train", progress),
            loss_fn=loss_fn,
            optimiser=optimiser,
            scaler=grad_scaler,
            device=device,
            metrics=train_metrics,
            clip_grad_norm=gradient_clipping,
        )

        val_stats = validate_epoch(
            model=model,
            loader=get_tqdm_bar(val_loader, epoch, epochs, "Train", progress),
            loss_fn=loss_fn,
            metrics=val_metrics,
            device=device,
        )

        scheduler.step(val_stats["loss_total"])  # epoch param is deprecated

        logger.info(
            "Epoch %d/%d: Avg Train Loss: %.4f, Avg Val Loss: %.4f, Mean IoU: %.4f, Training time: %s, Validation time: %s",
            epoch+1,
            epochs,
            train_stats["loss_total"],
            val_stats["loss_total"],
            val_stats.get(monitor_metric, 0.0),
            str(datetime.timedelta(seconds=int(train_stats["elapsed_time"]))),
            str(datetime.timedelta(seconds=int(val_stats["elapsed_time"])))
        )

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
                num_classes=num_classes
            )

            # ensure checkpoints directory exists
            ckpt_dir = Path(directory)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            model_name = getattr(model, 'name', model.__class__.__name__)

            # update best_vloss and best_score based on monitor_metric
            if val_stats["loss_total"] < best_vloss:
                best_vloss = val_stats["loss_total"]
                
            metric_value = val_stats.get(monitor_metric, None)
            if metric_value is not None and metric_value > best_score:
                best_score = metric_value
                save_ckpt(checkpoint, str(ckpt_dir / f"{model_name}-{epochs}_best.pth"))
            save_ckpt(checkpoint, str(ckpt_dir / f"{model_name}-{epochs}_current.pth"))

        if reporter is not None:
            lr_value = None
            if optimiser is not None and hasattr(optimiser, "param_groups") and optimiser.param_groups:
                lr_value = float(optimiser.param_groups[0].get("lr", 0.0))
            reporter.log_epoch(
                epoch=epoch + 1,
                epochs=epochs,
                train_stats=train_stats,
                val_stats=val_stats,
                lr=lr_value,
            )

        torch.cuda.empty_cache()  # clear cache

        loss_stop_policy(best_vloss)
        if loss_stop_policy.early_stop:
            logger.info("No improvement - terminating.")
            break


@click.group()
def cli():
    """leaf-segmentation"""
    pass


def build_reporter(report_name: str, report_dir: str, report_every: int, **metadata):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = report_name or f"{metadata['model'].name}-{timestamp}"
    report_path = Path(report_dir) / run_name
    reporter = SemanticTrainingReporter(
        output_dir=report_path,
        monitor_metric="mean_iou",
        plot_every=max(1, int(report_every)),
        append=metadata['resume'] is not None
    )
    reporter.write_metadata(metadata)
    return reporter

@cli.command()
@click.option("--model", type=str, required=True, help="Name of the model to use.")
@click.option("--encoder", type=str, required=True, help="Name of the encoder to use")
@click.option("--dataset", type=str, required=True, help="Name of the dataset to use")
@click.option("--num_classes", type=int, required=True, help="Number of semantic classes")
@click.option("--batch_size", type=int, default=8)
@click.option("--num_workers", type=int, default=4)
@click.option("--lr", type=float, default=1e-3, help="Learning rate")
@click.option("--epochs", type=int, default=100, help="Number of epochs to train on.")
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu",
              help="Device to train the model on. Default is 'cuda' if available")
@click.option("--resume", type=click.Path(exists=True), default=None,
              help="Resume from a prior checkpoint")
@click.option("--use_amp", is_flag=True)
@click.option("--progress/--no-progress",default=None,help="Enable/disable tqdm progress bars (default: auto based on TTY).",)
@click.option("--no_report", is_flag=True)
@click.option("-o", "--out", type=click.Path(exists=True), default="checkpoints",help="output directory")
def train(model, encoder, dataset, num_classes, batch_size, num_workers, lr, epochs, device, resume, use_amp, progress, no_report, out):
    # prepare dataloaders
    train_loader, val_loader = get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, num_classes=num_classes)
    model = setup_model(model, encoder, dataset, num_classes, device)

    # suggested by - https://arxiv.org/pdf/1206.5533
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.999], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=3)
    loss_fn = CEDiceLoss(ce_weight=0.5, dice_weight=0.5)

    progress_enabled = resolve_progress_flag(progress)

    # build reporter
    reporter = None
    if not no_report:
        reporter = build_reporter(
            report_name=f"{model.__class__.__name__.lower()}-{encoder}-{dataset}-{epochs}-report", 
            report_dir=out, 
            report_every=1,
            model = model.name,
            encoder = encoder,
            dataset = dataset,
            num_classes = num_classes,
            batch_size = batch_size,
            num_workers = num_workers,
            lr = lr,
            epochs = epochs,
            device = str(device),
            resume = resume,
            use_amp = bool(use_amp),
        )

    fit(
        model=model,
        loss_fn=loss_fn,
        optimiser=optimiser,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        device=device,
        num_classes=num_classes,
        use_amp=use_amp,
        gradient_clipping=0.1,
        monitor_metric="mean_iou",
        resume=resume,
        progress=progress_enabled,
        reporter=reporter,
        directory=out
    )


@cli.command()
@click.option("--model", type=str, required=True, help="Name of the model to use.")
@click.option("--encoder", type=str, required=True, help="Name of the encoder to use")
@click.option("--dataset", type=str, required=True, help="Name of the dataset to use")
@click.option("--num_classes", type=int, required=True, help="Number of semantic classes")
@click.option("--batch_size", type=int, default=8)
@click.option("--num_workers", type=int, default=4)
@click.option("--lr", type=float, default=1e-4, help="Base learning rate for finetuning")
@click.option("--epochs", type=int, default=50, help="Number of epochs to train on.")
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu",
              help="Device to train the model on. Default is 'cuda' if available")
@click.option("--ckpt", type=click.Path(exists=True), required=True,
              help="Path to a pretrained checkpoint (model_state/state_dict/model)")
@click.option("--freeze_encoder", "freeze_encoder_flag", is_flag=True, help="Freeze encoder for the entire finetuning run")
@click.option("--freeze_epochs", type=int, default=0,
              help="Freeze encoder for N epochs, then unfreeze for the remaining epochs")
@click.option("--encoder_lr", type=float, default=None, help="Encoder LR (default: lr * 0.1)")
@click.option("--decoder_lr", type=float, default=None, help="Decoder/head LR (default: lr)")
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--strict_load", is_flag=True, help="Require exact key/shape match when loading")
@click.option("--use_amp", is_flag=True)
@click.option("--progress/--no-progress",default=None,help="Enable/disable tqdm progress bars (default: auto based on TTY).",)
def finetune(
    model,
    encoder,
    dataset,
    num_classes,
    batch_size,
    num_workers,
    lr,
    epochs,
    device,
    ckpt,
    freeze_encoder_flag,
    freeze_epochs,
    encoder_lr,
    decoder_lr,
    weight_decay,
    strict_load,
    use_amp,
    progress
):
    train_loader, val_loader = get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, num_classes=num_classes)
    model = setup_model(model, encoder, dataset, num_classes, device)
    model.name = f"{model.name}_fn"

    progress_enabled = resolve_progress_flag(progress)

    # load pretrained weights (strict or shape-matched)
    load_pretrained_weights(model, ckpt, device=device, strict_load=strict_load)

    loss_fn = CEDiceLoss(ce_weight=0.5, dice_weight=0.5)

    base_lr = lr
    if encoder_lr is None:
        encoder_lr = base_lr * 0.1
    if decoder_lr is None:
        decoder_lr = base_lr

    # stage 1: optional frozen encoder
    remaining_epochs = epochs
    if freeze_epochs > 0:
        freeze_epochs = min(freeze_epochs, epochs)
        freeze_encoder_flag = True

        if freeze_encoder_flag:
            freeze_encoder(model)

        optimiser = build_finetune_optimiser(
            model=model,
            encoder_lr=encoder_lr,
            decoder_lr=decoder_lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=3)

        fit(
            model=model,
            loss_fn=loss_fn,
            optimiser=optimiser,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=freeze_epochs,
            device=device,
            num_classes=num_classes,
            use_amp=use_amp,
            gradient_clipping=0.1,
            monitor_metric="Mean IoU",
            start_epoch=0,
            save=False,
            progress=progress_enabled
        )
        remaining_epochs = epochs - freeze_epochs
        if remaining_epochs <= 0:
            return

        unfreeze_encoder(model)

    # stage 2 (or single stage): full finetuning
    if freeze_encoder_flag and freeze_epochs == 0:
        freeze_encoder(model)

    optimiser = build_finetune_optimiser(
        model=model,
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=3)

    start_epoch = epochs - remaining_epochs
    fit(
        model=model,
        loss_fn=loss_fn,
        optimiser=optimiser,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        device=device,
        num_classes=num_classes,
        use_amp=use_amp,
        gradient_clipping=0.1,
        monitor_metric="mean_iou",
        start_epoch=start_epoch,
        progress=progress_enabled
    )

# CLI wrapper
if __name__ == "__main__":
    cli()


# def _add_arguments(args: list):
#     def wrap(func):
#         for arg in args:
#             func = arg(func)
#         return func
#     return wrap


# _common_options = [
#     click.option("--model", type=str, required=True, help="Name of the model to use."),
#     click.option("--encoder", type=str, required=True, help="Name of the encoder to use"),
#     click.option("--batch_size", type=int, default=8),
#     click.option("--num_workers", type=int, default=4),
#     click.option("--epochs", type=int, default=100, help="Number of epochs to train on."),
#     click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu",
#               help="Device to train the model on. Default is 'cuda' if available"),
#     click.option("--use_amp", is_flag=True, help="Enable mixed-precision FP16"),
# ]
