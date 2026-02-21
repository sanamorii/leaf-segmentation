from dataclasses import dataclass
import json
import torch
import numpy as np
import albumentations as A
import logging

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from typing import List, Literal, Tuple, Optional
from pathlib import Path
from PIL import Image
from glob import glob

from leaf_seg.dataset.templates import SemanticDatasetSpec, SplitSpec
from leaf_seg.dataset.utils import TRAIN_TFMS, VAL_TFMS, get_dataset_spec, get_split_spec


logger = logging.getLogger(__name__)


class PlantDreamerData(Dataset):
    """
    PlantDreamer dataset format; depth, gt, mask
    Take an array of image_paths and mask_paths - this should be aggregated outside this function.
    """
    def __init__(self, image_paths, mask_paths, transforms=None):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        # self.n_classes = n_classes
        # self.colors_to_class = class_colors

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]))

        # TODO: add some functionality to account for rgb
        if len(mask.shape) > 2:
            raise RuntimeError("Mask is not monochrome, aborting")
        # mask = rgb_to_class(mask, class_colors=self.colors_to_class) 

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image'].float()
            mask = augmented['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask


def scan_pairs(root: Path, image_dir: str, mask_dir: str, ext: str) -> list[tuple[str, str]]:
    img_dir = root / image_dir
    msk_dir = root / mask_dir
    imgs = sorted (img_dir.glob(f"*.{ext}"))
    msks = sorted (msk_dir.glob(f"*.{ext}"))

    # match by filename
    msk_map = {p.name: p for p in msks}
    pairs = []
    for p in imgs:
        if p.name in msk_map:
            pairs.append((str(p), str(msk_map[p.name])))
    return pairs

def load_split_file_pairs(root: Path, image_dir: str, mask_dir: str, filelist: Path) -> list[tuple[str, str]]:
    img_dir = root / image_dir
    msk_dir = root / mask_dir
    names = [ln.strip() for ln in filelist.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [(str(img_dir / n), str(msk_dir / n)) for n in names]

def resolve_pairs(spec: SemanticDatasetSpec, split: SplitSpec, shuffle: bool = True) -> tuple[list[tuple[str,str]], list[tuple[str,str]] | None]:

    all_pairs = scan_pairs(spec.root, spec.image_dir, spec.mask_dir, spec.ext)

    if split.kind ==  "none":
        return all_pairs, None
    
    if split.kind ==  "files":
        if not split.train_files or not split.val_files:
            raise ValueError("file split requires train_files and val_files")
        train_pairs = load_split_file_pairs(spec.root, spec.image_dir, spec.mask_dir, split.train_files)
        val_pairs = load_split_file_pairs(spec.root, spec.image_dir, spec.mask_dir, split.val_files)
        return train_pairs, val_pairs

    if split.kind == "ratio":
        if not (0.0 < split.train_ratio < 1.0) or not (0.0 < split.val_ratio < 1.0):
            raise ValueError("ratios must be in (0,1)")
        
        if abs((split.train_ratio + split.val_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio must equal 1.0")
        
        train_pairs, val_pairs = train_test_split(
            all_pairs, test_size=split.val_ratio, random_state=split.seed, shuffle=shuffle
        )
        return train_pairs, val_pairs
    
    raise ValueError(f"Unknown split kind: {split.kind}")


def build_dataset(
    dataset_id: str,
    registry_path: str | Path,
    *,
    train_transforms: Optional[A.Compose] = None,
    val_transforms: Optional[A.Compose] = None,
    image_size: tuple[int, int] = (512, 512),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    shuffle: bool = True,
) -> tuple[DataLoader, Optional[DataLoader], SemanticDatasetSpec, SplitSpec]:
    spec = get_dataset_spec(dataset_id, registry_path)
    split = get_split_spec(dataset_id, registry_path)

    if spec.task != "semantic": # TODO: implement instance dataloading
        raise NotImplementedError("instance not available")
    
    if train_transforms is None:
        train_transforms = TRAIN_TFMS(image_size, mean, std)
    if val_transforms is None:
        val_transforms = VAL_TFMS(image_size, mean, std)
    
    train_pairs, val_pairs = resolve_pairs(spec, split, shuffle)

    trn_img, trn_msk = zip(*train_pairs)
    train_ds = PlantDreamerData(trn_img, trn_msk, transforms=train_transforms)

    val_ds: Dataset | None = None
    if val_pairs is not None:
        val_img, val_msk = zip(*val_pairs)
        val_ds = PlantDreamerData(val_img, val_msk, transforms=val_transforms)

    logger.info("Dataset=%s root=%s task=%s", spec.name, spec.root, spec.task)
    logger.info("Split kind=%s train=%d val=%s", split.kind, len(train_ds), (len(val_pairs) if val_pairs else None))

    return train_ds, val_ds, spec, split



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
) -> tuple[DataLoader, Optional[DataLoader], SemanticDatasetSpec, SplitSpec]:

    train_ds, val_ds, spec, split = build_dataset(
        dataset_id=dataset_id,
        registry_path=registry_path,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        image_size=image_size,
        mean=mean, std=std,
        shuffle=shuffle,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    val_loader: DataLoader | None = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            drop_last=False,
        )

    logger.info("Dataset=%s root=%s task=%s", spec.name, spec.root, spec.task)
    logger.info("Split kind=%s train=%d val=%s", split.kind, len(train_ds), (len(val_ds) if val_ds else None))

    return train_loader, val_loader, spec, split


# basic smoke test
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=str, default="datasets.yaml")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true")
    args = ap.parse_args()

    train_loader, val_loader, spec, split = build_dataloaders(
        dataset_id=args.dataset,
        registry_path=args.registry,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # iterate one batch to validate
    x, y = next(iter(train_loader))
    print("Train batch:", x.shape, y.shape)
    if val_loader is not None:
        x, y = next(iter(val_loader))
        print("Val batch:", x.shape, y.shape)