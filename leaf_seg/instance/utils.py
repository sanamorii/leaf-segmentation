"""Shared instance segmentation utilities (COCO format helpers, mask ops)."""

import numpy as np
import torch
from pycocotools import mask as mask_utils


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def boxes_xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] boxes to COCO [x, y, w, h] format."""
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    w = np.clip(x2 - x1, a_min=0, a_max=None)
    h = np.clip(y2 - y1, a_min=0, a_max=None)
    return np.stack([x1, y1, w, h], axis=1)


def mask_to_rle(binary_mask: np.ndarray):
    """Encode a binary HxW mask to COCO RLE format.

    pycocotools expects Fortran-ordered arrays.
    """
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def resize_masks_to_image(masks: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize predicted masks to match original image dimensions."""
    if masks.shape[-2:] == (out_h, out_w):
        return masks
    t = torch.from_numpy(masks).unsqueeze(1).float()
    t = torch.nn.functional.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return t[:, 0].cpu().numpy()


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def evaluate_image(pred_masks, pred_labels, gt_masks, gt_labels, iou_thresh):
    """Match predictions to ground truth for a single image.

    Returns (tp, fp, fn, matched_ious).
    """
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


def extract_predictions(output: dict, score_thresh: float):
    """Filter model output by score threshold, binarise masks.

    Returns (masks, labels, boxes) as lists of numpy arrays.
    """
    pred_masks = to_numpy(output["masks"])
    pred_masks = pred_masks[:, 0] if pred_masks.ndim == 4 else pred_masks
    pred_labels = to_numpy(output["labels"]).tolist()
    pred_scores = to_numpy(output["scores"]).tolist()
    pred_boxes = to_numpy(output["boxes"]).tolist()

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

    return filtered_masks, filtered_labels, filtered_boxes
