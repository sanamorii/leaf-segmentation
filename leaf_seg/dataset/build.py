import yaml
import logging
from pathlib import Path
from typing import Optional, Any, Tuple

import albumentations as A
from torch.utils.data import Dataset, DataLoader

from leaf_seg.dataset.plantdreamer_instance import coco_collate_fn
from leaf_seg.dataset.plantdreamer_semantic import build_dataset as build_pd_semantic
from leaf_seg.dataset.plantdreamer_instance import build_dataset as build_pd_instance
from leaf_seg.dataset.templates import DatasetSpec, InstanceDatasetSpec, SemanticDatasetSpec

logger = logging.getLogger(__name__)

def load_registry(registry_path: str | Path) -> dict:
    registry_path = Path(registry_path)

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry YAML not found: {registry_path}")

    with registry_path.open(mode="r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)

    if not isinstance(reg, dict):
        raise ValueError("Registry YAML must be a mapping (dict) at the top level")

    return reg


def get_dataset_spec(dataset_id: str, registry_path: str | Path) -> SemanticDatasetSpec | InstanceDatasetSpec:
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
    
    num_classes = cfg.get("num_classes")
    if num_classes is None:
        raise ValueError(f"Dataset '{dataset_id}' missing required key: root")

    task = cfg.get("task")
    if task not in ("semantic", "instance"):
        raise ValueError(f"Dataset '{dataset_id}' has invalid task '{task}'")
    
    train_set = cfg.get("train_set")
    if train_set is None:
        raise ValueError(f"Dataset '{dataset_id}' missing required key: train_set")
    
    val_set = cfg.get("val_set")
    if val_set is None:
        raise ValueError(f"Dataset '{dataset_id}' missing required key: val_set")
    
    labels = cfg.get("labels")
    if labels is None:
        raise ValueError(f"Dataset '{dataset_id}' missing required key: val_set")

    manifest = cfg.get("manifest", None)
    if manifest is not None: manifest = Path(manifest)

    values = {
        "name":dataset_id,
        "root":Path(root),
        "task":task,
        "num_classes": int(num_classes),
        "train_set":Path(train_set),
        "val_set":Path(val_set),
        "manifest":manifest,
    }
    
    if task == "semantic":
        return SemanticDatasetSpec(
            **values,
            image_dir=cfg.get("image_dir", "gt"),
            mask_dir=cfg.get("mask_dir", "masks"),
            ext=cfg.get("ext", "png"),
        )

    else:  # instance

        return InstanceDatasetSpec(
            **values,
            image_dir=cfg.get("image_dir", "gt"),
            remap=cfg.get("remap", True),
            filter_empty=cfg.get("filter_empty", True),
        )

def build_dataloaders(
    dataset_id: str,
    registry_path: str | Path,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    train_transforms: Optional[A.Compose] = None,
    val_transforms: Optional[A.Compose] = None,
    image_size: tuple[int, int] = (512, 512),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    shuffle: bool = True,
    drop_last: bool = True,
) -> tuple[DataLoader, DataLoader, DatasetSpec]:
    
    spec = get_dataset_spec(dataset_id, registry_path)

    if spec.task == "instance":
        collate_fn = coco_collate_fn
        train_ds, val_ds = build_pd_instance(
            spec=spec,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            image_size=image_size,
        )
    if spec.task == "semantic":
        collate_fn = None
        train_ds, val_ds = build_pd_semantic(
            spec=spec,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            image_size=image_size,
            mean=mean, std=std,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    logger.info("Dataset=%s root=%s task=%s", spec.name, spec.root, spec.task)
    logger.info("Images=%d train=%d val=%s", (len(train_ds)+len(val_ds)), len(train_ds), len(val_ds))


    return train_loader, val_loader, spec