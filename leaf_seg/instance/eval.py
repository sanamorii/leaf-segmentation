"""Instance segmentation evaluation runner."""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from pycocotools.cocoeval import COCOeval

from leaf_seg.common.config import InstanceEvalConfig
from leaf_seg.common.eval import load_instance_model, print_report, save_json_results
from leaf_seg.common.verbose import suppress_stout
from leaf_seg.common.vis import (
    add_title,
    load_raw_rgb,
    make_grid,
    make_labeled_montage,
    masks_to_boxes,
    render_instance_overlay,
    tensor_to_rgb,
)
from leaf_seg.dataset.build import build_dataloaders
from leaf_seg.dataset.plantdreamer_instance import LeafCoco, coco_collate_fn
from leaf_seg.instance.utils import (
    to_numpy,
    boxes_xyxy_to_xywh,
    mask_to_rle,
    resize_masks_to_image,
    evaluate_image,
    extract_predictions,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_coco(
    model: torch.nn.Module,
    dataset: LeafCoco,
    loader: DataLoader,
    device: str,
    score_thresh: float = 0.5,
    max_dets_per_image: int = 100,
    iou_types: tuple[str, ...] = ("bbox", "segm"),
    progress: bool = True,
) -> dict:
    """Run COCO-standard evaluation (AP/AR metrics).

    Returns dict with keys like segm_AP, segm_AP50, bbox_AP, etc.
    """
    model.eval()
    coco_results = []

    contig_to_cat_id = {
        i + 1: int(cat_id)
        for i, cat_id in enumerate(getattr(dataset, "cat_ids", []))
    }

    it = tqdm(loader, desc="COCO Eval") if progress else loader

    for images, targets in it:
        images_dev = [img.to(device) for img in images]
        outputs = model(images_dev)

        for img, out, tgt in zip(images_dev, outputs, targets):
            scores = to_numpy(out["scores"])
            keep = scores >= score_thresh
            if keep.sum() == 0:
                continue

            idx = np.argsort(scores[keep])[::-1]
            if idx.shape[0] > max_dets_per_image:
                idx = idx[:max_dets_per_image]

            boxes = to_numpy(out["boxes"])[keep][idx].astype(np.float32, copy=True)
            labels = to_numpy(out["labels"])[keep][idx]
            scores = scores[keep][idx]

            masks = None
            if "masks" in out and "segm" in iou_types:
                masks = to_numpy(out["masks"])[keep][idx]
                masks = masks[:, 0, :, :]

            image_id = int(to_numpy(tgt["image_id"]).reshape(-1)[0])
            input_h, input_w = int(img.shape[-2]), int(img.shape[-1])
            img_meta = dataset.coco.imgs.get(image_id, {})
            orig_h = int(img_meta.get("height", input_h))
            orig_w = int(img_meta.get("width", input_w))

            sx = float(orig_w) / float(max(1, input_w))
            sy = float(orig_h) / float(max(1, input_h))
            if sx != 1.0 or sy != 1.0:
                boxes[:, [0, 2]] *= sx
                boxes[:, [1, 3]] *= sy
                boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, float(orig_w))
                boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, float(orig_h))

            if masks is not None and masks.shape[-2:] != (orig_h, orig_w):
                masks = resize_masks_to_image(masks, orig_h, orig_w)

            boxes_xywh = boxes_xyxy_to_xywh(boxes)

            for i in range(boxes_xywh.shape[0]):
                res = {
                    "image_id": image_id,
                    "category_id": int(contig_to_cat_id.get(int(labels[i]), int(labels[i]))),
                    "bbox": [float(x) for x in boxes_xywh[i]],
                    "score": float(scores[i]),
                }
                if masks is not None:
                    bin_mask = (masks[i] >= 0.5).astype(np.uint8)
                    res["segmentation"] = mask_to_rle(bin_mask)
                coco_results.append(res)

    results = {}

    if len(coco_results) == 0:
        for t in iou_types:
            results.update({
                f"{t}_AP": 0.0, f"{t}_AP50": 0.0, f"{t}_AP75": 0.0,
                f"{t}_APs": 0.0, f"{t}_APm": 0.0, f"{t}_APl": 0.0,
                f"{t}_AR1": 0.0, f"{t}_AR10": 0.0, f"{t}_AR100": 0.0,
            })
        return results

    coco_gt = dataset.coco
    eval_img_ids = [int(x) for x in dataset.img_ids]

    with suppress_stout(True):
        coco_dt = coco_gt.loadRes(coco_results)

    for iou_type in iou_types:
        with suppress_stout(True):
            coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
            coco_eval.params.useCats = 1
            coco_eval.params.maxDets = [1, 10, 100]
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
        results[f"{iou_type}_AR1"] = float(stats[6])
        results[f"{iou_type}_AR10"] = float(stats[7])
        results[f"{iou_type}_AR100"] = float(stats[8])

    return results


@torch.no_grad()
def evaluate_matching(
    model: torch.nn.Module,
    dataset: LeafCoco,
    loader: DataLoader,
    device: str,
    score_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    progress: bool = True,
) -> dict:
    """Run per-image greedy matching evaluation (precision, recall, F1, mean IoU)."""
    model.eval()

    total_tp = total_fp = total_fn = 0
    all_ious = []

    it = tqdm(loader, desc="Matching Eval") if progress else loader

    for images, targets in it:
        images_dev = [img.to(device) for img in images]
        outputs = model(images_dev)

        for out, tgt in zip(outputs, targets):
            pred_masks, pred_labels, _ = extract_predictions(out, score_thresh)

            gt_masks_np = tgt["masks"].detach().cpu().numpy()
            gt_labels = tgt["labels"].detach().cpu().numpy().tolist()

            gt_masks = [gm.astype(np.uint8) for gm in gt_masks_np]
            gt_labels_filtered = [int(gl) for gl in gt_labels]

            tp, fp, fn, ious = evaluate_image(
                pred_masks, pred_labels,
                gt_masks, gt_labels_filtered,
                iou_thresh,
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            all_ious.extend(ious)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean_iou,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "num_matches": len(all_ious),
    }


def run(cfg: InstanceEvalConfig, registry_path: str = "data/datasets.yaml"):
    """Entry point for ``leaf-seg eval instance``."""

    model = load_instance_model(
        checkpoint=cfg.checkpoint,
        num_classes=cfg.num_classes,
        device=cfg.device,
    )

    _, val_loader, spec = build_dataloaders(
        dataset_id=cfg.dataset,
        registry_path=registry_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=False,
    )

    # unwrap to get the base LeafCoco dataset
    ds = val_loader.dataset
    while isinstance(ds, Subset):
        ds = ds.dataset

    coco_results = evaluate_coco(
        model=model,
        dataset=ds,
        loader=val_loader,
        device=cfg.device,
        score_thresh=cfg.score_thresh,
    )

    matching_results = evaluate_matching(
        model=model,
        dataset=ds,
        loader=val_loader,
        device=cfg.device,
        score_thresh=cfg.score_thresh,
        iou_thresh=cfg.iou_thresh,
    )

    all_results = {**coco_results, **matching_results}

    title = f"INSTANCE EVAL — Mask R-CNN on {cfg.dataset}"
    print_report(title, all_results, output_dir=cfg.output)
    save_json_results(all_results, cfg.output)

    if cfg.save_vis:
        save_visualisations(
            model=model,
            dataset=val_loader.dataset,
            num_samples=cfg.num_vis_samples,
            device=cfg.device,
            output_dir=cfg.output,
            score_thresh=cfg.score_thresh,
            dataset_name=f"Mask R-CNN on {cfg.dataset}",
        )

    return all_results


@torch.no_grad()
def save_visualisations(
    model: torch.nn.Module,
    dataset,
    num_samples: int,
    device: str,
    output_dir: str | Path,
    score_thresh: float = 0.5,
    dataset_name: str = "",
) -> None:
    """Save instance prediction montages (Image | GT Masks | Predicted Masks)
    for a subset of the validation set, similar to YOLO's val visualisations.

    Each sample gets a labelled side-by-side montage saved individually,
    and a combined results grid is saved at the end.
    """
    model.eval()
    output_dir = Path(output_dir) / "vis"
    (output_dir / "montage").mkdir(parents=True, exist_ok=True)
    (output_dir / "pred_only").mkdir(parents=True, exist_ok=True)

    # Unwrap Subset to reach the base dataset
    base_ds = dataset
    index_map: list[int] | None = None
    if isinstance(base_ds, Subset):
        index_map = list(base_ds.indices)
        base_ds = base_ds.dataset

    rng = np.random.default_rng(42)
    n = min(num_samples, len(dataset))
    idxs = rng.permutation(len(dataset))[:n].tolist()

    montages: list[Image.Image] = []

    for j, ds_idx in enumerate(idxs):
        sample = dataset[ds_idx]
        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            img_t, target = sample[0], sample[1]
        else:
            img_t, target = sample, None

        # Forward pass (Mask R-CNN expects list of images)
        preds = model([img_t.to(device)])
        pred = preds[0] if isinstance(preds, (list, tuple)) else preds

        # Extract filtered predictions
        pred_masks, pred_labels, pred_boxes = extract_predictions(pred, score_thresh)

        # Get visualisation image
        h, w = img_t.shape[-2], img_t.shape[-1]
        raw_idx = index_map[ds_idx] if index_map is not None else ds_idx
        img_rgb = load_raw_rgb(base_ds, raw_idx, out_hw=(h, w))
        if img_rgb is None:
            img_rgb = tensor_to_rgb(img_t)

        # Prediction overlay
        pred_vis = render_instance_overlay(img_rgb, pred_masks, pred_labels, pred_boxes)
        pred_pil = Image.fromarray(pred_vis, mode="RGB")

        # Save prediction-only overlay
        pred_pil.save(output_dir / "pred_only" / f"idx{ds_idx:05d}_pred.png")

        base_img = Image.fromarray(img_rgb, mode="RGB")
        panels: list[tuple[str, Image.Image]] = [("Image", base_img)]

        # Ground truth overlay (if available)
        if target is not None and isinstance(target, dict) and "masks" in target:
            gt_masks_t = target["masks"]
            if torch.is_tensor(gt_masks_t):
                gt_masks_np = gt_masks_t.detach().cpu().numpy()
            else:
                gt_masks_np = np.asarray(gt_masks_t)
            if gt_masks_np.ndim == 4 and gt_masks_np.shape[1] == 1:
                gt_masks_np = gt_masks_np[:, 0]

            gt_labels_t = target.get("labels")
            if gt_labels_t is not None:
                gt_labels = (gt_labels_t.detach().cpu().numpy().tolist()
                             if torch.is_tensor(gt_labels_t) else list(gt_labels_t))
            else:
                gt_labels = [1] * gt_masks_np.shape[0]

            gt_masks_list = [(m > 0).astype(np.uint8) for m in gt_masks_np]
            gt_boxes = masks_to_boxes(gt_masks_list)

            gt_vis = render_instance_overlay(img_rgb, gt_masks_list, gt_labels, gt_boxes)
            panels.append(("Ground Truth", Image.fromarray(gt_vis, mode="RGB")))

        panels.append(("Prediction", pred_pil))

        montage = make_labeled_montage(panels)
        montage.save(output_dir / "montage" / f"idx{ds_idx:05d}_montage.png")
        montages.append(montage)

    # Combined grid
    if montages:
        grid = make_grid(montages, ncols=2, pad=10)
        title = f"Instance Eval — {dataset_name}" if dataset_name else "Instance Eval"
        grid = add_title(grid, title)
        grid.save(output_dir / "results.png")
        logger.info("Saved %d visualisation montages to %s", len(montages), output_dir)
