import os
import json
import torch
import numpy as np
import albumentations as A
import logging

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional
from pathlib import Path
from PIL import Image
from glob import glob

from .utils import rgb_to_class

logger = logging.getLogger(__name__)

CLASS_COLORS = {
    0: (0, 0, 0),       # Background (black)
    1: (0, 255, 0),     # Leaf (green)
    2: (255, 165, 0),   # Pot (orange)
    3: (139, 69, 19),   # Soil (brown)
    4: (157, 0, 255),   # Stem (purple)
}
COLOR_TO_CLASS = {
    (0, 0, 0): 0,  # Background
    (1, 1, 1): 1,  # Leaf
    (2, 2, 2): 2,  # Pot
    (3, 3, 3): 3,  # Soil
    # (4, 4, 4): 4   # Stem
}

CLASSES = {
    "default":{
        (0, 0, 0): 0,  # Background
        (1, 1, 1): 1,  # Leaf
        (2, 2, 2): 2,  # Pot
        (3, 3, 3): 3,  # Soil
        (4, 4, 4): 4   # Stem
    },
    "reduced":{
        (0, 0, 0): 0,  # Background
        (1, 1, 1): 1,  # Leaf
        (2, 2, 2): 2,  # Pot
        (3, 3, 3): 3,  # Stem
    },
    "wheat":{
        (0,0,0): 0,
        (1, 1, 1): 1,  # Head
        (2, 2, 2): 2,  # Leaf
        (3, 3, 3): 3,  # Pot
        (4, 4, 4): 4   # Stem

    }
}


def collect_all_data(base_dir: str, ext: str = "png") -> List[Tuple[str, str]]:
    """Collect (image, mask) pairs under a base directory.

    Expects a structure like base_dir/<subdir>/gt/*.png and base_dir/<subdir>/mask/*.png
    Or a structure like base_dir/gt/*.png and base_dir/mask/*.png

    Returns a list of (image_path, mask_path) tuples (strings). Does not raise on
    empty datasets but logs a warning.
    """
    base = Path(base_dir)
    if not base.exists():
        logger.warning("Base directory does not exist: %s", base_dir)
        return []

    pairs: List[Tuple[str, str]] = []

    def collect_from_dirs(gt_dir: Path, mask_dir: Path):
        """Helper to collect pairs from a single gt/mask directory pair."""
        sub_pairs = []
        if not gt_dir.exists() or not mask_dir.exists():
            return sub_pairs

        images = sorted([str(p) for p in gt_dir.glob(f"*.{ext}")])
        masks = sorted([str(p) for p in mask_dir.glob(f"*.{ext}")])

        # assuming 1-to-1 filename match
        if len(images) != len(masks):
            logger.warning("Mismatched counts in %s: images=%d masks=%d", gt_dir.parent, len(images), len(masks))
            # continue to next subdir rather than asserting
            return sub_pairs

        for img_path, mask_path in zip(images, masks):
            if Path(img_path).name != Path(mask_path).name:
                logger.warning("Filename mismatch: %s vs %s", img_path, mask_path)
                continue
            sub_pairs.append((img_path, mask_path))
        return sub_pairs

    # case 1: flat structure (base_dir/gt and base_dir/mask)
    flat_gt = base / "gt"
    flat_mask = base / "mask"
    pairs.extend(collect_from_dirs(flat_gt, flat_mask))

    # case 2: nested subdirectories (base_dir/<subdir>/gt, mask) 
    for subdir in base.iterdir():
        if not subdir.is_dir():
            continue
        gt_dir = subdir / "gt"
        mask_dir = subdir / "mask"
        pairs.extend(collect_from_dirs(gt_dir, mask_dir))

    if len(pairs) == 0:
        logger.warning("No image/mask pairs found under %s", base_dir)

    return pairs


def get_dataloader(
    dataset: str,  # name of the dataset
    batch_size: int,
    num_workers: int,
    num_classes: int,
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
        "bean_semantic": "./data/bean_semantic",
        "wheat_semantic": "./data/wheat_semantic",
        "kale_semantic": "./data/kale_semantic",
        "mint_semantic": "./data/mint_semantic",
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

    if dataset in ["tomato"]:
        raise NotImplementedError(f"dataset {dataset} not available")

    # individual species support: 'bean', 'wheat', 'kale', 'mint', 'tomato'
    if dataset in species_dirs:
        species_base = species_dirs[dataset]
        all_pairs = collect_all_data(species_base, ext=ext)
        if len(all_pairs) == 0:
            raise RuntimeError(f"No data found for species '{dataset}' in {species_base} for extension {ext}")

        train_paths, val_paths = train_test_split(
            all_pairs, test_size=0.2, random_state=seed, shuffle=shuffle
        )
        train_imgs, train_masks = zip(*train_paths)
        val_imgs, val_masks = zip(*val_paths)
        train_ds = PlantDreamerData(list(train_imgs), list(train_masks), transforms=train_transforms, n_classes=num_classes)
        val_ds = PlantDreamerData(list(val_imgs), list(val_masks), transforms=val_transforms, n_classes=num_classes)

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


class PlantDreamerData(Dataset):
    """
    PlantDreamer dataset format; depth, gt, mask
    Take an array of image_paths and mask_paths - this should be aggregated outside this function.
    """
    def __init__(self, image_paths, mask_paths, n_classes, transforms=None):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.n_classes = n_classes
        # self.colors_to_class = class_colors

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]))

        # TODO: add some functionality to account for rgb
        if len(mask.shape) > 2:
            raise RuntimeError("Mask is not monochrome, aborting")
        if len(np.unique(mask)) > self.n_classes:
            raise RuntimeError(f"Number of classes in mask is not equal to reported no. of classes.\nViolating mask: {self.mask_paths[idx]}")

        # mask = rgb_to_class(mask, class_colors=self.colors_to_class) 

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image'].float()
            mask = augmented['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask


# class PlantDreamer(Dataset):
#     """
#     PlantDreamer dataset format; depth, gt, mask
#     Take an filepath to image_dir (correspondong to gt) and mask_dir (correspondong to mask)
#     """
#     def __init__(self, image_dir:str, mask_dir:str, class_colors, transforms=None) -> None:
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transforms = transforms
#         self.images = list(sorted(os.listdir(image_dir)))
#         self.masks = list(sorted(os.listdir(mask_dir)))
#         self.colors_to_class = class_colors

#         super().__init__()

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, index):
#         img_path = os.path.join(self.image_dir, self.images[index])
#         mask_path = os.path.join(self.mask_dir, self.masks[index])

#         image = np.array(Image.open(img_path).convert("RGB"))
#         mask = np.array(Image.open(mask_path).convert("RGB"))
#         mask = rgb_to_class(mask, class_colors=self.colors_to_class)  # class mask from RGB

#         if self.transforms:
#             augmented = self.transforms(image=image, mask=mask)
#             image = augmented["image"].float()
#             mask = augmented["mask"].long()
#         else:
#             image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
#             mask = torch.from_numpy(mask).long()
#         return image, mask
