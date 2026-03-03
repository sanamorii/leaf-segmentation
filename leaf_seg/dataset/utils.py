
from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import yaml

from leaf_seg.dataset.templates import InstanceDatasetSpec, SemanticDatasetSpec



def rgb_to_class(mask, class_colors):
    mask = np.array(mask)
    class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for color, class_id in class_colors.items():
        match = np.all(mask == color, axis=-1)
        class_mask[match] = class_id
    return class_mask

def decode_mask(mask, class_colors):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        color_mask[mask == class_id] = color

    return color_mask

def overlay(image, color_mask, alpha=0.5):
    return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)



def _resolve_path(base_dir: Path, value: str | Path | None) -> Optional[Path]:
    if value is None or base_dir is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p)

