
from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import yaml

from leaf_seg.dataset.templates import InstanceDatasetSpec, SemanticDatasetSpec, SplitSpec



def TRAIN_TFMS(
    image_size: tuple[int, int] = (512, 512),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    h, w = image_size
    return A.Compose(
        [
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def VAL_TFMS(
    image_size: tuple[int, int] = (512, 512),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    h, w = image_size
    return A.Compose(
        [
            A.Resize(h, w),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )




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


def load_registry(registry_path: str | Path) -> dict:
    registry_path = Path(registry_path)

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry YAML not found: {registry_path}")

    with registry_path.open(mode="r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)

    if not isinstance(reg, dict):
        raise ValueError("Registry YAML must be a mapping (dict) at the top level")

    return reg


def get_dataset_spec(dataset_id: str, registry_path: str | Path) -> SemanticDatasetSpec:
    reg = load_registry(registry_path)

    if dataset_id not in reg:
        keys = sorted(k for k in reg.keys() if k != "splits")
        raise KeyError(f"Unknown dataset id '{dataset_id}'. Available: {keys}")

    cfg = reg[dataset_id]
    if not isinstance(cfg, dict):
        raise ValueError(f"Dataset entry '{dataset_id}' must map to a dict.")

    root = cfg.get("root")
    if root is None:
        raise ValueError(f"Dataset '{dataset_id}' missing required key: root")

    task = cfg.get("task")
    if task not in ("semantic", "instance"):
        raise ValueError(f"Dataset '{dataset_id}' has invalid task '{task}'")
    
    if task == "semantic":
        return SemanticDatasetSpec(
            name=dataset_id,
            root=Path(root),
            task=task,
            image_dir=cfg.get("image_dir", "gt"),
            mask_dir=cfg.get("mask_dir", "masks"),
            ext=cfg.get("ext", "png"),
        )

    else:  # instance
        return InstanceDatasetSpec(
            name=dataset_id,
            root=Path(root),
            image_dir=cfg.get("image_dir", "gt"),
            ann=Path(cfg.get("ann", "coco.json")),
            remap=cfg.get("remap", True),
            filter_empty=cfg.get("filter_empty", True),
        )

def get_split_spec(dataset_id: str, registry_path: str | Path) -> SplitSpec:
    reg = load_registry(registry_path)
    ds_cfg = reg.get(dataset_id)
    if not isinstance(ds_cfg, dict):
        raise KeyError(f"Unknown dataset id '{dataset_id}'.")

    split_cfg = ds_cfg.get("split")

    # default split if absent
    if split_cfg is None:
        return SplitSpec(kind="ratio", train_ratio=0.8, val_ratio=0.2, seed=42)

    if not isinstance(split_cfg, dict):
        raise ValueError(f"split for '{dataset_id}' must be a dict")

    kind = split_cfg.get("kind", "ratio")
    if kind not in ("ratio", "files", "none"):
        raise ValueError(f"Invalid split kind '{kind}' for '{dataset_id}'")

    if kind == "none":
        return SplitSpec(kind="none")

    seed = int(split_cfg.get("seed", 42))

    if kind == "ratio":
        tr = float(split_cfg.get("train_ratio", 0.8))
        vr = float(split_cfg.get("val_ratio", 0.2))
        if abs((tr + vr) - 1.0) > 1e-6:
            raise ValueError(f"train_ratio + val_ratio must equal 1.0 (got {tr}+{vr})")
        return SplitSpec(kind="ratio", seed=seed, train_ratio=tr, val_ratio=vr)

    if kind == "files":
        train_files = _resolve_path(ds_cfg.get("root"),split_cfg.get("train_files"))
        val_files = _resolve_path(ds_cfg.get("root"),split_cfg.get("val_files"))
        if train_files is None or val_files is None:
            raise ValueError("files split requires train_files and val_files")
        return SplitSpec(kind="files", seed=seed, train_files=train_files, val_files=val_files)