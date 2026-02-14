import os
import random
import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from tqdm import tqdm

from dataset.plantdreamer_instance import LeafCoco
from models.maskrcnn_torch import get_model as get_maskrcnn
from models.utils import load_ckpt

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _validate_dataset_root(dataset_root: Path):
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise click.ClickException(f"not a valid folder: {dataset_root}")
    if not (dataset_root / "gt").is_dir():
        raise click.ClickException(f"missing 'gt' folder under: {dataset_root}")
    if not (dataset_root / "coco.json").is_file():
        raise click.ClickException(f"missing 'coco.json' under: {dataset_root}")


def _resize_sample(img_t: torch.Tensor, masks_t: torch.Tensor, resize: tuple[int, int] | None):
    if resize is None:
        return img_t, masks_t

    img_t = F.interpolate(
        img_t.unsqueeze(0),
        size=resize,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    if masks_t.numel() == 0:
        masks_t = masks_t.new_zeros((0, resize[0], resize[1]))
    else:
        masks_t = F.interpolate(
            masks_t.unsqueeze(0).float(),
            size=resize,
            mode="nearest",
        ).squeeze(0).to(torch.uint8)

    return img_t, masks_t


def mask_iou(a: np.ndarray, b: np.ndarray):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def evaluate_image(pred_masks, pred_labels, gt_masks, gt_labels, iou_thresh):
    matched_gt = set()
    matched_pred = set()
    ious = []

    for p_idx, (pm, pl) in enumerate(zip(pred_masks, pred_labels)):
        best_iou = 0.0
        best_gt = None
        for g_idx, (gm, gl) in enumerate(zip(gt_masks, gt_labels)):
            if g_idx in matched_gt:
                continue
            if pl != gl:
                continue
            iou = mask_iou(pm, gm)
            if iou > best_iou:
                best_iou = iou
                best_gt = g_idx
        if best_gt is not None and best_iou >= iou_thresh:
            matched_gt.add(best_gt)
            matched_pred.add(p_idx)
            ious.append(best_iou)

    tp = len(matched_pred)
    fp = len(pred_masks) - tp
    fn = len(gt_masks) - len(matched_gt)

    return tp, fp, fn, ious


def _mask_to_box(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return [0, 0, 0, 0]
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return [x0, y0, x1, y1]


def _boxes_from_masks(masks: list[np.ndarray]) -> list[list[int]]:
    return [_mask_to_box(mask) for mask in masks]


def _draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: list[int],
    font_scale: float = 0.4,
    thickness: int = 1,
):
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def render_overlay(
    image: np.ndarray,
    masks: list[np.ndarray],
    labels: list[int],
    boxes: list[list[int]],
    label_map: dict[int, str] | None,
    alpha: float = 0.4,
):
    overlay = image.copy()
    for mask, label, box in zip(masks, labels, boxes):
        color = [random.randint(0, 255) for _ in range(3)]
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask.astype(bool)] = color
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)

        x0, y0, x1, y1 = [int(v) for v in box]
        if x1 > x0 and y1 > y0:
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 1)
            name = label_map.get(label, f"Class {label}") if label_map else f"Class {label}"
            _draw_label(overlay, name, x0, max(10, y0 - 3), color)
    return overlay


def _add_panel_title(image: np.ndarray, title: str):
    _draw_label(image, title, 10, 20, [255, 255, 255], font_scale=0.5, thickness=1)


def _merge_panels(panels: list[np.ndarray]) -> np.ndarray:
    heights = [p.shape[0] for p in panels]
    widths = [p.shape[1] for p in panels]
    if len(set(heights)) != 1:
        max_h = max(heights)
        resized = []
        for p in panels:
            if p.shape[0] == max_h:
                resized.append(p)
                continue
            new_w = int(p.shape[1] * (max_h / p.shape[0]))
            resized.append(cv2.resize(p, (new_w, max_h), interpolation=cv2.INTER_AREA))
        panels = resized
    return np.concatenate(panels, axis=1)


def save_visuals(
    save_dir,
    basename,
    image,
    pred_masks,
    pred_labels,
    pred_boxes,
    gt_masks,
    gt_labels,
    gt_boxes,
    label_map,
):
    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(image).save(f"{save_dir}/{basename}_image.png")
    pred_overlay = None
    gt_overlay = None

    if pred_masks is not None:
        pred_overlay = render_overlay(
            image,
            pred_masks,
            pred_labels,
            pred_boxes,
            label_map,
        )
        Image.fromarray(pred_overlay).save(f"{save_dir}/{basename}_pred.png")
    if gt_masks is not None:
        gt_overlay = render_overlay(
            image,
            gt_masks,
            gt_labels,
            gt_boxes,
            label_map,
        )
        Image.fromarray(gt_overlay).save(f"{save_dir}/{basename}_gt.png")

    if pred_overlay is not None and gt_overlay is not None:
        orig_panel = image.copy()
        _add_panel_title(orig_panel, "original")
        _add_panel_title(gt_overlay, "ground truth")
        _add_panel_title(pred_overlay, "prediction")
        merged = _merge_panels([orig_panel, gt_overlay, pred_overlay])
        Image.fromarray(merged).save(f"{save_dir}/{basename}_merged.png")


def load_model(checkpoint_path: str, num_classes: int, device: str):
    model = get_maskrcnn(num_classes=num_classes)
    ckpt = load_ckpt(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def evaluate_dataset(
    model,
    dataset_root: str,
    save_dir: str,
    device: str,
    resize=None,
    score_thresh=0.5,
    iou_thresh=0.5,
    verbosity=0,
):
    base = Path(dataset_root)
    _validate_dataset_root(base)

    ds = LeafCoco(
        image_dir=str(base / "gt"),
        annotation_file=str(base / "coco.json"),
        transforms=None,
        remap=True,
        filter_empty=True,
    )

    label_map = {}
    for cat in ds.coco.loadCats(ds.coco.getCatIds()):
        cat_id = int(cat["id"])
        label = ds.cat_id_to_contiguous.get(cat_id)
        if label is not None:
            label_map[int(label)] = str(cat.get("name", f"Class {label}"))

    total_tp = total_fp = total_fn = 0
    all_ious = []

    loader = tqdm(range(len(ds)), desc="Evaluating") if verbosity > 0 else range(len(ds))

    with torch.no_grad():
        for idx in loader:
            img_t, target = ds[idx]

            img_t, masks_t = _resize_sample(img_t, target["masks"], resize)
            target = dict(target)
            target["masks"] = masks_t

            image_id = int(target["image_id"].item())
            img_info = ds.coco.loadImgs([image_id])[0]
            basename = Path(img_info["file_name"]).stem

            img_tensor = img_t.to(device)
            outputs = model([img_tensor])
            output = outputs[0]

            pred_masks = output["masks"].detach().cpu().numpy()
            pred_masks = pred_masks[:, 0] if pred_masks.ndim == 4 else pred_masks
            pred_labels = output["labels"].detach().cpu().numpy().tolist()
            pred_scores = output["scores"].detach().cpu().numpy().tolist()
            pred_boxes = output["boxes"].detach().cpu().numpy().tolist()

            filtered_masks = []
            filtered_labels = []
            filtered_boxes = []
            for pm, pl, ps, pb in zip(pred_masks, pred_labels, pred_scores, pred_boxes):
                if ps < score_thresh:
                    continue
                if pl == 0:
                    continue
                filtered_masks.append((pm >= 0.5).astype(np.uint8))
                filtered_labels.append(int(pl))
                filtered_boxes.append([int(v) for v in pb])

            gt_masks_np = target["masks"].detach().cpu().numpy()
            gt_labels = target["labels"].detach().cpu().numpy().tolist()

            gt_masks = []
            gt_labels_filtered = []
            for gm, gl in zip(gt_masks_np, gt_labels):
                gt_masks.append(gm.astype(np.uint8))
                gt_labels_filtered.append(int(gl))
            gt_boxes = _boxes_from_masks(gt_masks)

            tp, fp, fn, ious = evaluate_image(
                filtered_masks,
                filtered_labels,
                gt_masks,
                gt_labels_filtered,
                iou_thresh,
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            all_ious.extend(ious)

            if save_dir is not None:
                img_np = (img_t.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(
                    np.uint8
                )
                save_visuals(
                    save_dir,
                    basename,
                    img_np,
                    filtered_masks,
                    filtered_labels,
                    filtered_boxes,
                    gt_masks,
                    gt_labels_filtered,
                    gt_boxes,
                    label_map,
                )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "num_matches": len(all_ious),
    }


def print_report(results: dict, output: str | None):
    lines = []
    lines.append("RESULTS FOR Mask R-CNN")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Precision : {results['precision']:.4f}")
    lines.append(f"Recall    : {results['recall']:.4f}")
    lines.append(f"F1 Score  : {results['f1']:.4f}")
    lines.append(f"Mean IoU  : {results['mean_iou']:.4f}")
    lines.append("")
    lines.append(f"TP: {results['tp']}  FP: {results['fp']}  FN: {results['fn']}")
    lines.append(f"Matched instances: {results['num_matches']}")

    msg = "\n".join(lines)
    print(msg)

    if output is not None:
        os.makedirs(output, exist_ok=True)
        path = Path(output) / "results.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(msg)
        print(f"Saved report to: {path}")


@click.group()
def cli():
    """Instance segmentation evaluation (Mask R-CNN)"""
    pass


@cli.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--num_classes", required=True, type=int)
@click.option(
    "--dataset",
    required=True,
    type=click.Path(exists=True),
    help="PlantDreamer COCO dataset root (expects coco.json + gt/)",
)
@click.option("--output", default="results_instance", type=str)
@click.option("--device", default="cuda")
@click.option("--resize", default=None, nargs=2, type=int)
@click.option("--score_thresh", default=0.5, type=float)
@click.option("--iou_thresh", default=0.5, type=float)
@click.option("--verbosity", "-v", count=True, help="Increase verbosity.")
def evaluate(
    checkpoint,
    num_classes,
    dataset,
    output,
    device,
    resize,
    score_thresh,
    iou_thresh,
    verbosity,
):
    model = load_model(checkpoint, num_classes, device)

    metrics = evaluate_dataset(
        model=model,
        dataset_root=dataset,
        save_dir=f"{output}/predictions",
        device=device,
        resize=tuple(resize) if resize is not None else None,
        score_thresh=score_thresh,
        iou_thresh=iou_thresh,
        verbosity=verbosity,
    )

    print_report(metrics, output=output)
    click.echo("Evaluation complete.")


if __name__ == "__main__":
    cli()
