"""Shared visualization utilities for eval-time prediction saving.

Consolidates helpers from batch_sem_infer.py and batch_inst_infer.py so that
``leaf-seg eval semantic --save-vis`` and ``leaf-seg eval instance --save-vis``
can produce labelled montages (Image | GT | Prediction) alongside metrics.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ── colour palettes ──────────────────────────────────────────────────────────

# Semantic palette (RGBA) – index == class id
SEMANTIC_PALETTE: list[tuple[int, int, int, int]] = [
    (0, 0, 0, 255),       # 0  background
    (0, 255, 0, 255),     # 1
    (255, 0, 0, 255),     # 2
    (0, 0, 255, 255),     # 3
    (255, 255, 0, 255),   # 4
    (255, 0, 255, 255),   # 5
    (0, 255, 255, 255),   # 6
    (128, 128, 0, 255),   # 7
    (128, 0, 128, 255),   # 8
]

# Instance palette (RGB) – cycled per-instance
INSTANCE_PALETTE: list[list[int]] = [
    [0, 0, 255],
    [0, 200, 0],
    [255, 102, 0],
    [0, 170, 255],
    [255, 0, 180],
    [200, 200, 0],
    [140, 0, 255],
    [0, 120, 120],
    [120, 120, 0],
    [80, 80, 180],
]

# Default ImageNet normalisation (used for de-normalising tensors)
VIS_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
VIS_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


# ── tensor → numpy helpers ───────────────────────────────────────────────────

def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Clip a float [0,1] or uint8 [0,255] HWC image to uint8."""
    if img.dtype == np.uint8:
        return img
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def tensor_to_rgb(image_t: torch.Tensor) -> np.ndarray:
    """Convert a [C,H,W] tensor (possibly normalised) to uint8 HWC RGB.

    Uses conservative per-channel rescaling when the value range does not
    match [0,1] or [0,255], falling back to ImageNet de-normalisation when
    the range looks like normalised data.
    """
    if image_t.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(image_t.shape)}")

    x = image_t.detach().cpu().float().permute(1, 2, 0).numpy()  # HWC

    if x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    elif x.shape[2] > 3:
        x = x[:, :, :3]

    mn, mx = float(x.min()), float(x.max())

    if 0.0 <= mn and mx <= 1.0:
        pass  # already OK
    elif 0.0 <= mn and mx <= 255.0:
        x = x / 255.0
    elif mn >= -5.0 and mx <= 5.0:
        # likely ImageNet-normalised
        ch = x.shape[2]
        mean = VIS_MEAN[:ch].reshape(1, 1, ch)
        std = VIS_STD[:ch].reshape(1, 1, ch)
        x = x * std + mean
    else:
        # robust per-channel rescale
        x = x.copy()
        for c in range(x.shape[2]):
            lo, hi = float(x[:, :, c].min()), float(x[:, :, c].max())
            if hi - lo > 1e-6:
                x[:, :, c] = (x[:, :, c] - lo) / (hi - lo)
            else:
                x[:, :, c] = 0.0

    return to_uint8_rgb(np.clip(x, 0.0, 1.0))


def load_raw_rgb(ds: Any, ds_idx: int, out_hw: tuple[int, int]) -> np.ndarray | None:
    """Best-effort raw RGB loader from a dataset's ``image_paths`` attribute."""
    h, w = out_hw
    for attr in ("image_paths", "img_paths", "images"):
        paths = getattr(ds, attr, None)
        if paths is None:
            continue
        try:
            p = paths[ds_idx]
        except (IndexError, KeyError):
            continue
        try:
            with Image.open(p) as im0:
                im = im0.convert("RGB")
                if im.size != (w, h):
                    im = im.resize((w, h), resample=Image.BILINEAR)
                return np.asarray(im, dtype=np.uint8)
        except Exception:
            pass
    return None


# ── semantic visualisation ───────────────────────────────────────────────────

def colorize_mask(
    mask: np.ndarray,
    palette: list[tuple[int, int, int, int]] | None = None,
) -> Image.Image:
    """Convert an HxW integer mask to an RGBA PIL image (background transparent)."""
    palette = palette or SEMANTIC_PALETTE
    h, w = mask.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_id in np.unique(mask):
        if int(cls_id) <= 0:
            continue
        color = palette[int(cls_id) % len(palette)]
        rgba[mask == cls_id] = color
    return Image.fromarray(rgba, mode="RGBA")


def overlay_mask(image_rgb: np.ndarray, mask_rgba: Image.Image, alpha: float = 0.45) -> Image.Image:
    """Composite an RGBA mask onto an RGB image with the given alpha."""
    base = Image.fromarray(image_rgb, mode="RGB").convert("RGBA")
    m = mask_rgba.copy()
    arr = np.array(m).astype(np.float32)
    arr[..., 3] = arr[..., 3] * alpha
    m = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    return Image.alpha_composite(base, m).convert("RGB")


# ── instance visualisation ───────────────────────────────────────────────────

def render_instance_overlay(
    image: np.ndarray,
    masks: list[np.ndarray],
    labels: list[int],
    boxes: list[list[int]],
    palette: list[list[int]] | None = None,
    alpha: float = 0.4,
    show_boxes: bool = True,
    show_labels: bool = True,
    label_map: dict[int, str] | None = None,
) -> np.ndarray:
    """OpenCV-based instance overlay.  Expects & returns RGB uint8."""
    palette = palette or INSTANCE_PALETTE
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    overlay = image.copy()

    for i, (mask, label, box) in enumerate(zip(masks, labels, boxes)):
        color = palette[(int(label) - 1) % len(palette)] if int(label) > 0 else palette[i % len(palette)]
        mask_bool = mask.astype(bool)
        if np.any(mask_bool):
            overlay[mask_bool] = (
                (1 - alpha) * overlay[mask_bool].astype(np.float32)
                + alpha * np.array(color, dtype=np.float32)
            ).astype(np.uint8)

        x0, y0, x1, y1 = (int(v) for v in box)
        if show_boxes and x1 > x0 and y1 > y0:
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)

        if show_labels and x1 > x0 and y1 > y0:
            text = label_map.get(int(label), str(label)) if label_map else str(label)
            cv2.putText(overlay, text, (x0, max(y0 - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return overlay


def masks_to_boxes(masks: list[np.ndarray]) -> list[list[int]]:
    """Compute tight bounding boxes from a list of binary masks."""
    boxes = []
    for m in masks:
        ys, xs = np.where(m)
        if ys.size == 0:
            boxes.append([0, 0, 0, 0])
        else:
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    return boxes


# ── montage / grid builders ─────────────────────────────────────────────────

def make_labeled_montage(
    panels: list[tuple[str, Image.Image]],
    pad: int = 8,
    label_h: int = 60,
    bg: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Horizontal montage with a label row above each panel."""
    if not panels:
        raise ValueError("No panels provided.")

    images = [im.convert("RGB") for _, im in panels]
    w0, h0 = images[0].size
    n = len(images)
    out_w = n * w0 + (n + 1) * pad
    out_h = h0 + label_h + 2 * pad
    out = Image.new("RGB", (out_w, out_h), color=bg)
    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 30)
    except Exception:
        font = ImageFont.load_default()

    y_img = pad + label_h
    draw.line((0, y_img - 1, out_w, y_img - 1), fill=(220, 220, 220), width=1)

    for i, (label, im) in enumerate(zip([n for n, _ in panels], images)):
        x = pad + i * (w0 + pad)
        if im.size != (w0, h0):
            im = im.resize((w0, h0), resample=Image.BILINEAR)
        out.paste(im, (x, y_img))

        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x + max(0, (w0 - tw) // 2)
        ty = pad + max(0, (label_h - th) // 2)
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

    return out


def make_grid(
    images: list[Image.Image],
    ncols: int = 4,
    pad: int = 8,
    bg: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Arrange PIL images into a grid."""
    if not images:
        raise ValueError("No images for grid.")
    w0, h0 = images[0].size
    ncols = max(1, ncols)
    nrows = math.ceil(len(images) / ncols)
    grid_w = ncols * w0 + (ncols + 1) * pad
    grid_h = nrows * h0 + (nrows + 1) * pad
    canvas = Image.new("RGB", (grid_w, grid_h), color=bg)
    for i, im in enumerate(images):
        r, c = divmod(i, ncols)
        x = pad + c * (w0 + pad)
        y = pad + r * (h0 + pad)
        canvas.paste(im.resize((w0, h0)) if im.size != (w0, h0) else im, (x, y))
    return canvas


def add_title(im: Image.Image, title: str, bar_h: int = 60) -> Image.Image:
    """Prepend a title bar to the top of an image."""
    w, h = im.size
    out = Image.new("RGB", (w, h + bar_h), color=(255, 255, 255))
    out.paste(im, (0, bar_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    draw.text((16, 8), title, fill=(0, 0, 0), font=font)
    return out
