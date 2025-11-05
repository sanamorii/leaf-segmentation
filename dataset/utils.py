import os
import cv2
from glob import glob
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from .bean import PlantDreamerAllBean, PlantDreamerBean
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def collect_all_data(base_dir: str, ext: str = "png") -> List[Tuple[str, str]]:
    """Collect (image, mask) pairs under a base directory.

    Expects a structure like base_dir/<subdir>/gt/*.png and base_dir/<subdir>/mask/*.png

    Returns a list of (image_path, mask_path) tuples (strings). Does not raise on
    empty datasets but logs a warning.
    """
    base = Path(base_dir)
    if not base.exists():
        logger.warning("Base directory does not exist: %s", base_dir)
        return []

    pairs: List[Tuple[str, str]] = []

    for subdir in base.iterdir():
        if not subdir.is_dir():
            continue
        gt_dir = subdir / "gt"
        mask_dir = subdir / "mask"

        if not gt_dir.exists() or not mask_dir.exists():
            continue

        images = sorted([str(p) for p in gt_dir.glob(f"*.{ext}")])
        masks = sorted([str(p) for p in mask_dir.glob(f"*.{ext}")])

        # assuming 1-to-1 filename match
        if len(images) != len(masks):
            logger.warning("Mismatched counts in %s: images=%d masks=%d", subdir, len(images), len(masks))
            # continue to next subdir rather than asserting
            continue

        for img_path, mask_path in zip(images, masks):
            if Path(img_path).name != Path(mask_path).name:
                logger.warning("Filename mismatch: %s vs %s", img_path, mask_path)
                continue
            pairs.append((img_path, mask_path))

    if len(pairs) == 0:
        logger.warning("No image/mask pairs found under %s", base_dir)

    return pairs



def decode_mask(mask, class_colors):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        color_mask[mask == class_id] = color

    return color_mask



def overlay(image, color_mask, alpha=0.5):
    return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)


def get_dataloader(
    dataset: str,  # name of the dataset
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
    shuffle: bool = True,
    base_dir: str = "./data/beans",
    image_size: int = 256,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ext: str = "png",
    seed: int = 42,
    drop_last: bool = False,
    train_transforms: Optional[A.Compose] = None,
    val_transforms: Optional[A.Compose] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Factory to create train/validation DataLoaders for supported dataset ids.

    Parameters:
    - dataset: identifier, currently supports 'bean01' and 'all'
    - base_dir: root data folder used for 'all' and bean-specific paths
    - image_size, mean, std: augmentation/resizing params
    - ext: image file extension
    - seed: random seed for train/val split
    - drop_last: pass to DataLoader for training
    - train_transforms / val_transforms: optional custom albumentations.Compose transforms

    Returns (train_loader, val_loader)
    """
    base_dir = str(base_dir)

    # mapping of species identifiers to data folders (can be adjusted)
    species_dirs = {
        "bean": "./data/beans",
        "wheat": "./data/wheat",
        "kale": "./data/kale",
        "mint": "./data/mint",
        "tomato": "./data/tomato",
    }

    # default augmentations if caller didn't provide custom ones
    if train_transforms is None:
        train_transforms = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    if val_transforms is None:
        val_transforms = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    # special small-bean split used previously
    if dataset == "bean01":
        train_ds = PlantDreamerBean(
            image_dir=os.path.join(base_dir, "bean0", "gt"),
            mask_dir=os.path.join(base_dir, "bean0", "mask"),
            transforms=train_transforms,
        )
        val_ds = PlantDreamerBean(
            image_dir=os.path.join(base_dir, "bean1", "gt"),
            mask_dir=os.path.join(base_dir, "bean1", "mask"),
            transforms=val_transforms,
        )

    # individual species support: 'bean', 'wheat', 'kale', 'mint', 'tomato'
    elif dataset in species_dirs:
        species_base = species_dirs[dataset]
        all_pairs = collect_all_data(species_base, ext=ext)
        if len(all_pairs) == 0:
            raise RuntimeError(f"No data found for species '{dataset}' in {species_base} for extension {ext}")

        train_paths, val_paths = train_test_split(
            all_pairs, test_size=0.2, random_state=seed, shuffle=shuffle
        )
        train_imgs, train_masks = zip(*train_paths)
        val_imgs, val_masks = zip(*val_paths)
        train_ds = PlantDreamerAllBean(list(train_imgs), list(train_masks), transforms=train_transforms)
        val_ds = PlantDreamerAllBean(list(val_imgs), list(val_masks), transforms=val_transforms)

    # combine all species listed in species_dirs
    elif dataset == "all":
        all_pairs = []
        for sp, path in species_dirs.items():
            all_pairs.extend(collect_all_data(path, ext=ext))

        if len(all_pairs) == 0:
            raise RuntimeError(f"No data found in any species directories: {list(species_dirs.values())}")

        train_paths, val_paths = train_test_split(
            all_pairs, test_size=0.2, random_state=seed, shuffle=shuffle
        )
        train_imgs, train_masks = zip(*train_paths)
        val_imgs, val_masks = zip(*val_paths)
        train_ds = PlantDreamerAllBean(list(train_imgs), list(train_masks), transforms=train_transforms)
        val_ds = PlantDreamerAllBean(list(val_imgs), list(val_masks), transforms=val_transforms)

    else:
        raise ValueError(f"Invalid dataset identifier: {dataset}")

    logger.info("Training dataset size: %d", len(train_ds))
    logger.info("Validation dataset size: %d", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader
