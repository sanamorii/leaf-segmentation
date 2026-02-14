
import datetime
import time
from pathlib import Path
import click

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

from dataset.plantdreamer_instance import get_dataloader
from models.maskrcnn_torch import get_model as get_maskrcnn
from models.utils import create_maskrcnn_ckpt, save_ckpt
from segmentation.reporter.instance import InstanceTrainingReporter
from segmentation.utils.verbose import get_tqdm_bar, resolve_progress_flag, suppress_stout

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _boxes_xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    # COCO expects [x, y, w, h]
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    w = np.clip(x2 - x1, a_min=0, a_max=None)
    h = np.clip(y2 - y1, a_min=0, a_max=None)
    return np.stack([x1, y1, w, h], axis=1)


def _unwrap_dataset(ds) -> Dataset:
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds

def _resolve_loader_dataset(loader) -> Dataset:
    """
    Resolve the underlying dataset from either a raw DataLoader or a tqdm-wrapped loader.
    """
    current = loader
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "dataset"):
            return current.dataset
        current = getattr(current, "iterable", None)
    raise AttributeError("Could not resolve `.dataset` from loader.")


def _subset_img_ids(ds) -> list[int]:
    """
    Resolve COCO image ids from a dataset that may be wrapped in nested Subset objects.
    """
    if isinstance(ds, Subset):
        parent_ids = _subset_img_ids(ds.dataset)
        return [int(parent_ids[int(i)]) for i in ds.indices]
    if hasattr(ds, "img_ids"):
        return [int(x) for x in ds.img_ids]
    raise AttributeError("Dataset must expose `img_ids` for subset COCO evaluation.")

def _mask_to_rle(binary_mask: np.ndarray):
    """
    COCO requires RLE. (pycocotools expects Fortran order).
    
    :param binary_mask: HxW {0,1} uint8/bool.
    :type binary_mask: np.ndarray
    """

    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    # pycocotools return bytes for counts; COCO json expects utf-8 strings
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def train_epoch(model : nn.Module, 
                loader : Iterable[tuple[torch.Tensor, torch.Tensor]], 
                optimiser : Optimizer, 
                device, 
                scaler : Optional[GradScaler] = None, 
                clip_grad_norm : Optional[float] = None
              ):
    start = time.time()
    model.train()
    running = {
        "loss_total": 0.0,
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_mask": 0.0,
    }
    n_batches = 0

    for image, target in loader:

        # send ground truth and annotation to device
        images = [img.to(device) for img in image]
        targets = [{k: v.to(device) for k, v in t.items()} for t in target] 

        optimiser.zero_grad(set_to_none=True)


        # amp fp16 (mixed precision) enabled
        if scaler is not None:
            device_type = 'cuda' if getattr(device, 'type', str(device)).startswith('cuda') else 'cpu'
            with torch.amp.autocast(device_type=device_type):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
        
            if clip_grad_norm is not None:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            scaler.step(optimiser)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimiser.step()

        # logging
        running["loss_total"] += float(loss.item())
        for k in running.keys():
            if k != "loss_total" and k in loss_dict:
                running[k] += float(loss_dict[k].item())

        if hasattr(loader, "set_postfix"):
            loader.set_postfix({
                "loss": float(loss.item()),
                "mask": float(loss_dict.get("loss_mask", 0.0).item() if "loss_mask" in loss_dict else 0.0),
                "cls": float(loss_dict.get("loss_classifier", 0.0).item() if "loss_classifier" in loss_dict else 0.0),
            })
        
        n_batches += 1
    
    for key in ("loss_total", "loss_classifier", "loss_box_reg", "loss_mask"):
        running[key] /= max(1, n_batches)
    running["elapsed_time"] = time.time() - start
    return running

@torch.no_grad()
def validate_epoch(
        model: nn.Module,
        loader: Iterable[tuple[torch.Tensor, torch.Tensor]], # validation set dataloader
        device: str,
        iou_types=("bbox","segm"),
        score_thresh: float= 0.0,
        max_dets_per_image: int = 100
):
    start = time.time()
    model.eval()
    coco_results = []
    running = {
        "loss_total": 0.0,
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_mask": 0.0,
    }

    dataset = _resolve_loader_dataset(loader)
    base_dataset = _unwrap_dataset(dataset)
    eval_img_ids = _subset_img_ids(dataset)
    contig_to_cat_id = {
        i + 1: int(cat_id)
        for i, cat_id in enumerate(getattr(base_dataset, "cat_ids", []))
    }
    
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            
            scores = _to_numpy(out["scores"])
            keep = scores >= score_thresh
            if keep.sum() == 0:
                continue
            
            idx = np.argsort(scores[keep])[::-1]
            if idx.shape[0] > max_dets_per_image:
                idx = idx[:max_dets_per_image]

            boxes = _to_numpy(out["boxes"])[keep][idx]
            labels = _to_numpy(out["labels"])[keep][idx]
            scores = scores[keep][idx]

            masks = None
            if "masks" in out and "segm" in iou_types:
                masks = _to_numpy(out["masks"])[keep][idx]
                masks = masks[:, 0, :, :] # [N, H, W]

            image_id = int(_to_numpy(tgt["image_id"]).reshape(-1)[0])
            boxes_xywh = _boxes_xyxy_to_xywh(boxes)

            for i in range(boxes_xywh.shape[0]):
                res = {
                    "image_id": image_id,
                    "category_id": int(contig_to_cat_id.get(int(labels[i]), int(labels[i]))),
                    "bbox": [float(x) for x in boxes_xywh[i]],
                    "score": float(scores[i])
                }

                if masks is not None:
                    bin_mask = (masks[i] >= 0.5).astype(np.uint8)
                    res["segmentation"] = _mask_to_rle(bin_mask)

                coco_results.append(res)

        running["loss_total"] = float(sum(outputs.values()).item())
        for k in running.keys():
            if k != "loss_total" and k in outputs:
                running[k] += float(outputs[k].item())


        if hasattr(loader, "set_postfix"):
            loader.set_postfix({"detections": len(coco_results)})
    
    results = {}

    # if no predictions: return zero
    if len(coco_results) == 0:
        for t in iou_types:
            results.update({
                f"{t}_AP": 0.0, f"{t}_AP50": 0.0, f"{t}_AP75": 0.0,
                f"{t}_APs": 0.0, f"{t}_APm": 0.0, f"{t}_APl": 0.0,
            })
        results["elapsed_time"] = time.time() - start
        return results
    
    coco_gt = base_dataset.coco
    with suppress_stout(True):
        coco_dt = coco_gt.loadRes(coco_results)

    for iou_type in iou_types:
        with suppress_stout(True): # suppress thing
            coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
            coco_eval.params.useCats = 1
            coco_eval.params.maxDets = [1, 10, 100] # coco standard
            coco_eval.params.imgIds = eval_img_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        results[f"{iou_type}_AP"] = float(stats[0])
        results[f"{iou_type}_AP50"] = float(stats[1])
        results[f"{iou_type}_AP75"] = float(stats[2])
        results[f"{iou_type}_APs"] = float(stats[3])
        results[f"{iou_type}_APm"] = float(stats[4])
        results[f"{iou_type}_APl"] = float(stats[5])

    results["elapsed_time"] = time.time() - start
    return results


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimiser: Optimizer,
    device,
    epochs: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    use_amp: bool = True,
    clip_grad_norm: Optional[float] = 1.0,
    out_dir: str = "results/maskrcnn",
    save_every: int = 1,
    metric_to_track: str = "segm_AP",
    higher_is_better: bool = True,
    val_score_thresh: float = 0.05,
    progress: bool = False,
    reporter: InstanceTrainingReporter | None = None,
):
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.to(device)

    scaler : GradScaler = GradScaler(enabled=(use_amp and device.type == "cuda"))
    out_dir : Path = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_metric = float("-inf") if higher_is_better else float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):

        train_stats = train_epoch(
            model=model,
            loader=get_tqdm_bar(train_loader, epoch - 1, epochs, "Train", progress),
            optimiser=optimiser,
            device=device,
            scaler=scaler if scaler.is_enabled() else None,
            clip_grad_norm=clip_grad_norm
        )
        
        val_stats = validate_epoch(
            model=model,
            loader=get_tqdm_bar(val_loader, epoch - 1, epochs, "Val", progress),
            device=device,
            iou_types=("bbox", "segm"),
            score_thresh=val_score_thresh,
            max_dets_per_image=100,
        )

        if scheduler is not None:
            scheduler.step(val_stats[metric_to_track])
        
        directory = Path(out_dir)
        directory.parent.mkdir(parents=True, exist_ok=True)

        current = float(val_stats.get(metric_to_track, 0.0))
        is_best = (current > best_metric) if higher_is_better else (current < best_metric)
        ckpt = create_maskrcnn_ckpt(
            model=model,
            optimiser=optimiser,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            train_stats=train_stats,
            val_stats=val_stats,
        )

        if is_best:
            best_metric = current
            best_epoch = epoch
            save_ckpt(ckpt, str(directory / f"{model.name}-{epochs}_best.pth"))

        if (epoch % save_every) == 0:
            save_ckpt(ckpt, str(directory / f"{model.name}-{epochs}_current.pth"))

        lr = optimiser.param_groups[0]["lr"]
        print(
            f"[epoch {epoch:03d}/{epochs}] "
            f"lr={lr:.2e} "
            f"train_loss={train_stats['loss_total']:.4f} "
            f"mask_loss={train_stats.get('loss_mask', 0.0):.4f} "
            f"val_segm_AP={val_stats.get('segm_AP', 0.0):.4f} "
            f"val_bbox_AP={val_stats.get('bbox_AP', 0.0):.4f} "
            f"{'(BEST)' if is_best else ''}"
        )
        if reporter is not None:
            reporter.log_epoch(
                epoch=epoch,
                epochs=epochs,
                train_stats=train_stats,
                val_stats=val_stats,
                lr=float(lr),
            )

    return {"best_metric": best_metric, "best_epoch": best_epoch}


def build_reporter(report_name: str, report_dir: str, report_every: int, **metadata):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = report_name or f"{metadata['model']}-{timestamp}"
    report_path = Path(report_dir) / run_name
    resume_path = metadata.get("resume")
    reporter = InstanceTrainingReporter(
        output_dir=report_path,
        monitor_metric="segm_AP",
        plot_every=max(1, int(report_every)),
        append=bool(resume_path),
    )
    reporter.write_metadata(metadata)
    return reporter




def setup_maskrcnn(num_classes: int, dataset: str, device: str) -> torch.nn.Module:
    model = get_maskrcnn(num_classes=num_classes)
    model.name = f"maskrcnn-{dataset}-instance"
    return model.to(torch.device(device))



@click.group()
def cli():
    """leaf-segmentation instance trainer (Mask R-CNN + COCOeval)"""
    pass


@cli.command()
@click.option("--dataset", type=str, required=True, help="Name of the dataset to use (passed into get_dataloader)")
@click.option("--num_classes", type=int, default=2, show_default=True, help="Includes background. Leaf-only => 2.")
@click.option("--batch_size", type=int, default=2, show_default=True)
@click.option("--num_workers", type=int, default=4, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--epochs", type=int, default=30, show_default=True)
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu")
@click.option("--resume", type=click.Path(exists=True), default=None)
@click.option("--use_amp", is_flag=True, help="Enable AMP (CUDA only).")
@click.option("--gradient_clipping", type=float, default=1.0, show_default=True)
@click.option("--val_score_thresh", type=float, default=0.05, show_default=True)
@click.option("--progress/--no-progress", default=None, help="Enable/disable tqdm progress bars (default: auto based on TTY).")
@click.option("--no_report", is_flag=True, help="Disable epoch reporter output.")
@click.option("--report_every", type=int, default=5, show_default=True, help="Write plots every N epochs.")
@click.option("-o", "--out", type=click.Path(), default="checkpoints/maskrcnn", show_default=True, help="Checkpoint/report output directory.")
def train(
    dataset: str,
    num_classes: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    epochs: int,
    device: str = "cuda",
    resume: str | None = None,
    use_amp: bool = True,
    gradient_clipping: float = 1.0,
    val_score_thresh: float = 0.05,
    progress: bool | None = None,
    no_report: bool = False,
    report_every: int = 5,
    out: str = "checkpoints/maskrcnn",
):
    progress_enabled = resolve_progress_flag(progress)
    train_loader, val_loader = get_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    model = setup_maskrcnn(num_classes=num_classes, dataset=dataset, device=device)
    model.run_id = f"{dataset}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=3
    )

    reporter = None
    if not no_report:
        reporter = build_reporter(
            report_name=f"maskrcnn-{dataset}-instance-report",
            report_dir=out,
            report_every=report_every,
            model=model.name,
            dataset=dataset,
            num_classes=num_classes,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            epochs=epochs,
            device=str(device),
            resume=resume,
            use_amp=bool(use_amp),
            gradient_clipping=gradient_clipping,
            val_score_thresh=val_score_thresh,
        )

    fit(
        model=model,
        optimiser=optimiser,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        device=device,
        use_amp=use_amp,
        out_dir=out,
        clip_grad_norm=gradient_clipping,
        val_score_thresh=val_score_thresh,
        progress=progress_enabled,
        reporter=reporter,
    )

if __name__ == "__main__":
    cli()
