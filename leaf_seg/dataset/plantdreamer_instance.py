from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import logging
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from leaf_seg.dataset.templates import InstanceDatasetSpec, SplitSpec
from leaf_seg.dataset.utils import TRAIN_TFMS, VAL_TFMS, get_dataset_spec, get_split_spec

logger = logging.getLogger(__name__)

def build_transforms(image_size: int | None, train: bool):
    transforms = []
    if image_size is not None:
        transforms.append(A.Resize(image_size, image_size))
    if train:
        transforms.append(A.HorizontalFlip(p=0.5))
        transforms.append(A.RandomBrightnessContrast(p=0.2))
    transforms.append(ToTensorV2())
    return A.Compose(transforms)

class LeafCoco(Dataset):
    """
    returns:
        image: FloatTensor [3,H,W] in [0,1]
        target: dict with keys:
            boxes (FloatTensor [N,4] xyxy)
            labels (Int64Tensor [N]) contiguous 1..K
            masks (UInt8Tensor [N,H,W])
            image_id (Int64Tensor [1])
            area (FloatTensor [N])
            iscrowd (Int64Tensor [N])
    """

    def __init__(
        self,
        image_dir,
        annotation_file,
        transforms=None,
        remap: bool = True,
        filter_empty: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.coco = COCO(annotation_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.filter_empty = filter_empty

        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_contiguous = {cid: i + 1 for i, cid in enumerate(self.cat_ids)}
        self.remap = remap

        self._normalize_iscrowd_annotations(annotation_file)

        if self.filter_empty:
            kept = []
            for img_id in self.img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=[img_id])
                if len(ann_ids) > 0:
                    kept.append(img_id)
            self.img_ids = kept

    def _normalize_iscrowd_annotations(self, annotation_file: str | Path) -> None:
        """
        Some CVAT/COCO exports mark all instances as `iscrowd=1`.
        COCOeval then treats all GT as crowd/ignore and returns AP = -1.
        """
        anns = self.coco.dataset.get("annotations", [])
        if not anns:
            return

        crowd_values = []
        for ann in anns:
            try:
                crowd_values.append(int(ann.get("iscrowd", 0)))
            except (TypeError, ValueError):
                crowd_values.append(0)

        if all(v == 1 for v in crowd_values):
            logger.warning(
                "All annotations in %s have iscrowd=1; converting to iscrowd=0 for training/eval.",
                annotation_file,
            )
            for ann in anns:
                ann["iscrowd"] = 0
            for ann in self.coco.anns.values():
                ann["iscrowd"] = 0

    def __len__(self):
        return len(self.img_ids)

    def _load_image(self, img_info: Dict[str, Any]) -> Image.Image:
        path = self.image_dir / img_info["file_name"]
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")

    def _ann_to_mask(self, ann: Dict[str, Any], height: int, width: int) -> np.ndarray:

        seg = ann.get("segmentation", None)
        if seg is None:
            return np.zeros((height, width), dtype=np.uint8)

        if isinstance(seg, list):
            rles = mask_utils.frPyObjects(seg, height, width)
            rle = mask_utils.merge(rles)
            m = mask_utils.decode(rle)
            return (m > 0).astype(np.uint8)

        if isinstance(seg, dict) and "counts" in seg and "size" in seg:
            # RLE can be compressed (counts as str/bytes) or uncompressed (counts as list).
            if isinstance(seg["counts"], list):
                rle = mask_utils.frPyObjects(seg, height, width)
                m = mask_utils.decode(rle)
                if m.ndim == 3:
                    m = m[:, :, 0]
                return (m > 0).astype(np.uint8)

            rle = seg
            if isinstance(rle["counts"], str):
                rle = dict(rle)
                rle["counts"] = rle["counts"].encode("ascii")
            m = mask_utils.decode(rle)
            if m.ndim == 3:
                m = m[:, :, 0]
            return (m > 0).astype(np.uint8)

        return np.zeros((height, width), dtype=np.uint8)

    @staticmethod
    def _xywh_to_xyxy(box: List[float]) -> List[float]:
        x, y, w, h = box
        return [x, y, x + w, y + h]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]

        pil_img = self._load_image(img_info)
        w, h = pil_img.size

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            bbox_xyxy = self._xywh_to_xyxy(ann["bbox"])
            x0, y0, x1, y1 = bbox_xyxy

            x0 = max(0.0, min(x0, w - 1.0))
            y0 = max(0.0, min(y0, h - 1.0))
            x1 = max(0.0, min(x1, w * 1.0))
            y1 = max(0.0, min(y1, h * 1.0))

            if x1 <= x0 or y1 <= y0:
                continue

            m = self._ann_to_mask(ann, h, w)
            if m.sum() == 0:
                continue

            cat_id = int(ann["category_id"])
            label = self.cat_id_to_contiguous[cat_id] if self.remap else cat_id

            boxes.append([x0, y0, x1, y1])
            labels.append(label)
            masks.append(m)
            areas.append(float(ann.get("area", m.sum())))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, h, w), dtype=torch.uint8)
            areas_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            masks_t = torch.from_numpy(np.stack(masks, axis=0)).to(torch.uint8)
            areas_t = torch.tensor(areas, dtype=torch.float32)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)

        img_t = torch.from_numpy(np.array(pil_img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": areas_t,
            "iscrowd": iscrowd_t,
        }

        if self.transforms is not None:
            img_t, target = self.transforms(img_t, target)

        return img_t, target


import json
import argparse
from collections import Counter

from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np



def coco_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def resolve_split(spec: InstanceDatasetSpec, split: SplitSpec, shuffle: bool = True):

    if split.kind == "none":
        ...

    if split.kind == "file":
        ...

    if split.kind == "ratio":

        if not (0.0 < split.train_ratio < 1.0) or not (0.0 < split.val_ratio < 1.0):
            raise ValueError("ratios must be in (0,1)")
        
        if abs((split.train_ratio + split.val_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio must equal 1.0")
        
        trn, val = split_coco(ann=spec.ann, train_path="coco_train.json", val_path="coco_val.json")
        

def build_dataloader(
    dataset_id: str,
    registry_path: str | Path,
    *,
    train_transforms: Optional[A.Compose] = None,
    val_transforms: Optional[A.Compose] = None,
    image_size: tuple[int, int] = (512, 512),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    shuffle: bool = True,
):
    ...
    # spec = get_dataset_spec(dataset_id, registry_path)
    # split = get_split_spec(dataset_id, registry_path)

    # if train_transforms is None:
    #     train_transforms = TRAIN_TFMS(image_size, mean, std)
    # if val_transforms is None:
    #     val_transforms = VAL_TFMS(image_size, mean, std)
    
    # train_pairs, val_pairs = resolve_pairs(spec, split, shuffle)

    # trn_img, trn_msk = zip(*train_pairs)
    # train_ds = PlantDreamerData(trn_img, trn_msk, transforms=train_transforms)

    # val_ds: Dataset | None = None
    # if val_pairs is not None:
    #     val_img, val_msk = zip(*val_pairs)
    #     val_ds = PlantDreamerData(val_img, val_msk, transforms=val_transforms)

    # logger.info("Dataset=%s root=%s task=%s", spec.name, spec.root, spec.task)
    # logger.info("Split kind=%s train=%d val=%s", split.kind, len(train_ds), (len(val_pairs) if val_pairs else None))

    # return train_ds, val_ds, spec, split


def get_dataloader(
    dataset: str,
    batch_size: int = 2,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    transforms=None,
    image_size: int | None = 512,
) -> Tuple[DataLoader, DataLoader]:
    
    dataset_dirs = {
        "bean_instance_synth": "data/bean_instance_synth/train",
        "bean_instance_real": "data/bean_instance_real/train",
    }

    base = Path(dataset_dirs[dataset])
    if (not base.exists()) or (not base.is_dir()) or (not (base / "gt").is_dir()) or (not (base / "coco.json").is_file()):
        logger.warning("Base directory does not exist: %s", dataset_dirs[dataset])
        return []
    

    ds = LeafCoco(
        image_dir=str(base / "gt"),
        annotation_file=str(base / "coco.json"),
        transforms=transforms,
        remap=True,
        filter_empty=True,
    )
    train_len = int(0.8 * len(ds))
    val_len = len(ds) - train_len

    train_ds, val_ds = random_split(ds, [train_len, val_len])

    # train_tfms = build_transforms(image_size=image_size, train=True)
    # val_tfms = build_transforms(image_size=image_size, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=coco_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=coco_collate_fn,
    )
    return train_loader, val_loader
