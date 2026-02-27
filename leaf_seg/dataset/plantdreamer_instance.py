from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from leaf_seg.dataset.templates import InstanceDatasetSpec
from leaf_seg.dataset.utils import get_dataset_spec

logger = logging.getLogger(__name__)


def TRAIN_TFMS(image_size: tuple[int, int] = None,) -> A.Compose:
    tfms = []
    if image_size is not None:
        h, w = image_size
        tfms.append(A.Resize(h, w))

    tfms += [
        # A.HorizontalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),
        ToTensorV2(),
    ]

    return A.Compose(
        tfms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=0.0,
            min_visibility=0.0,
            # clip=True,
        )
    )

def VAL_TFMS(image_size: Optional[tuple[int, int]] = None,) -> A.Compose:
    """
    Validation transforms: keep deterministic.
    No Normalize for torchvision Mask R-CNN.
    """
    tfms = []
    if image_size is not None:
        h, w = image_size
        tfms.append(A.Resize(h, w))

    tfms += [
        ToTensorV2(),
    ]

    return A.Compose(
        tfms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=0.0,
            min_visibility=0.0,
        ),
    )

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
        ann_file,
        transforms=None,
        remap: bool = True,
        filter_empty: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.coco : COCO = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transforms : A.Compose = transforms
        self.filter_empty = filter_empty

        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_contiguous = {cid: i + 1 for i, cid in enumerate(self.cat_ids)}
        self.remap = remap

        self._normalize_iscrowd_annotations(ann_file)

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

    @staticmethod
    def _xywh_to_xyxy(box: List[float]) -> List[float]:
        x, y, w, h = box
        return [x, y, x + w, y + h]
    
    @staticmethod
    def _clamp_xyxy(box: List[float], w: int, h: int) -> Optional[List[float]]:
        x0, y0, x1, y1 = box
        # Clamp to image bounds (allow x1==w, y1==h; torchvision can handle this)
        x0 = float(max(0.0, min(x0, float(w))))
        y0 = float(max(0.0, min(y0, float(h))))
        x1 = float(max(0.0, min(x1, float(w))))
        y1 = float(max(0.0, min(y1, float(h))))

        if x1 <= x0 or y1 <= y0:
            return None
        return [x0, y0, x1, y1]

    @staticmethod
    def _mask_to_box(mask: np.ndarray) -> list[float] | None:
        pos = np.where(mask > 0)
        if pos[0].size == 0 or pos[1].size == 0:
            return None
        xmin = float(np.min(pos[1]))
        xmax = float(np.max(pos[1]))
        ymin = float(np.min(pos[0]))
        ymax = float(np.max(pos[0]))
        if xmax <= xmin or ymax <= ymin:
            return None
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def _to_float01_image_tensor(img_t: torch.Tensor) -> torch.Tensor:

        if img_t.dtype == torch.uint8:
            return img_t.float().div(255.0)
        #if used ToTensorV2(always_apply=True) with float output or other custom conversion
        # if not torch.is_floating_point(img_t):
        img_t = img_t.float()
        # if it looks like [0,255] floats, scale down.
        if img_t.max().item() > 1.5:  #heuristic
            img_t = img_t.div(255.0)

        return img_t

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]

        pil_img = self._load_image(img_info)
        w, h = pil_img.size  # PIL: (W, H)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes_xyxy: List[List[float]] = []
        labels: List[int] = []
        masks: List[np.ndarray] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in anns:

            m = self.coco.annToMask(ann).astype(np.uint8) # [H, W] {0, 1}
            if m.sum() == 0:
                continue

            box = self._xywh_to_xyxy(ann["bbox"])
            box = self._clamp_xyxy(box, w=w, h=h)
            if box is None:
                continue

            cat_id = int(ann["category_id"])
            label = self.cat_id_to_contiguous[cat_id] if self.remap else cat_id

            boxes_xyxy.append(box)
            labels.append(label)
            masks.append(m)
            areas.append(float(ann.get("area", m.sum())))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        img_np = np.array(pil_img, dtype=np.uint8)

        if self.transforms is not None:
            transformed = self.transforms(
                image=img_np,
                masks=masks,
                bboxes=boxes_xyxy,
                labels=labels,
            )


            img_t = transformed["image"]
            img_t = self._to_float01_image_tensor(img_t)

            # if not torch.is_floating_point(img_t):
            #     img_t = img_t.float().div(255.0)
            # else:
            #     img_t = img_t.float()

            t_masks = transformed.get("masks", [])
            t_bboxes = transformed.get("bboxes", [])
            t_labels = transformed.get("labels", [])
            assert len(t_masks) == len(t_bboxes) == len(t_labels)

            out_h, out_w = int(img_t.shape[-2]), int(img_t.shape[-1])

            # filter out any degenerate boxes (albumentations can drop/clamp depending on params)
            new_boxes: List[List[float]] = []
            new_labels: List[int] = []
            new_masks: List[torch.Tensor] = []
            new_areas: List[float] = []
            new_iscrowd: List[int] = []

            for i, (mask_np, box, lab) in enumerate(zip(t_masks, t_bboxes, t_labels)):
                box = list(map(float, box))
                box = self._clamp_xyxy(box, w=out_w, h=out_h)
                if box is None:
                    continue
                
                # masks returned from albumentations correspond to original ordering;
                # when boxes get dropped, we should also drop masks by same index if possible.
                if i >= len(t_masks):
                    continue

                mask_arr = np.asarray(mask_np)
                if mask_arr.ndim > 2:
                    mask_arr = np.squeeze(mask_arr)
                mask_arr = (mask_arr > 0).astype(np.uint8)
                if mask_arr.sum() == 0:
                    continue

                # box_from_mask = self._mask_to_box(mask_arr)
                # if box_from_mask is None:
                #     continue

                new_masks.append(torch.from_numpy(mask_arr).to(torch.uint8))
                new_boxes.append(box)
                new_labels.append(int(lab))
                new_areas.append(float(mask_arr.sum()))
                new_iscrowd.append(int(iscrowd[i]))

            if len(new_boxes) == 0:
                boxes_t = torch.zeros((0, 4), dtype=torch.float32)
                labels_t = torch.zeros((0,), dtype=torch.int64)
                masks_t = torch.zeros((0, out_h, out_w), dtype=torch.uint8)
                areas_t = torch.zeros((0,), dtype=torch.float32)
                iscrowd_t = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes_t = torch.tensor(new_boxes, dtype=torch.float32)
                labels_t = torch.tensor(new_labels, dtype=torch.int64)
                masks_t = torch.stack(new_masks).to(torch.uint8)
                areas_t = torch.tensor(new_areas, dtype=torch.float32)
                iscrowd_t = torch.tensor(new_iscrowd, dtype=torch.int64)
        else:
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() # / 255.0
            img_t = self._to_float01_image_tensor(img_t)

            if len(boxes_xyxy) == 0:
                boxes_t = torch.zeros((0, 4), dtype=torch.float32)
                labels_t = torch.zeros((0,), dtype=torch.int64)
                masks_t = torch.zeros((0, h, w), dtype=torch.uint8)
                areas_t = torch.zeros((0,), dtype=torch.float32)
                iscrowd_t = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes_t = torch.tensor(boxes_xyxy, dtype=torch.float32)
                labels_t = torch.tensor(labels, dtype=torch.int64)
                masks_t = torch.from_numpy(np.stack(masks, axis=0)).to(torch.uint8)
                areas_t = torch.tensor(areas, dtype=torch.float32)
                iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)
                

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": areas_t,
            "iscrowd": iscrowd_t,
        }

        return img_t, target


def coco_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataset(
    dataset_id: str,
    registry_path: str | Path,
    *,
    train_transforms: Optional[A.Compose] = None,
    val_transforms: Optional[A.Compose] = None,
    image_size: tuple[int, int] = (512, 512),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),  # kept for API compatibility
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),   # kept for API compatibility
):
    spec = get_dataset_spec(dataset_id, registry_path)
    

    # NOTE: for some stupid reason pytorch's maskrcnn already performs transforms (normalisation, resizing)
    #     vision/torchvision/models/detection/faster_rcnn.py:281. do not transform here

    if train_transforms is None:
        train_transforms = TRAIN_TFMS(image_size=None)
    if val_transforms is None:
        val_transforms = VAL_TFMS(image_size=None)

    if not spec.train_set or not spec.val_set:
        raise ValueError("leafseg requires train_files and val_files requires")

    train_ds = LeafCoco(
        image_dir=str(spec.root / spec.image_dir), 
        ann_file=str(spec.root / spec.train_set), 
        remap=spec.remap,
        filter_empty=spec.filter_empty,
        transforms=train_transforms,
    )

    val_ds = LeafCoco(
        image_dir=str(spec.root / spec.image_dir), 
        ann_file=str(spec.root / spec.val_set), 
        remap=spec.remap,
        filter_empty=spec.filter_empty,
        transforms=val_transforms,
    )

    return train_ds, val_ds, spec

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
) -> tuple[DataLoader, DataLoader, InstanceDatasetSpec]:

    train_ds, val_ds, spec = build_dataset(
        dataset_id=dataset_id,
        registry_path=registry_path,
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
        collate_fn=coco_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
        collate_fn=coco_collate_fn,
    )

    return train_loader, val_loader, spec
