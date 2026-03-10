import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional
from pathlib import Path
from PIL import Image

from leaf_seg.dataset.templates import SemanticDatasetSpec

logger = logging.getLogger(__name__)

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

class PlantDreamerData(Dataset):
    """
    PlantDreamer dataset format; depth, gt, mask
    Take an array of image_paths and mask_paths - this should be aggregated outside this function.
    """
    def __init__(self, image_paths, mask_paths, transforms=None, remap=None):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.remap_lut = remap
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

        # if self.remap_lut is not None:
        #     mask = self.remap_lut[mask]

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image'].float()
            mask = augmented['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask

# TODO: another use for this...
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
    filelist = root / filelist
    names = [ln.strip() for ln in filelist.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [(str(img_dir / n), str(msk_dir / n)) for n in names]


def build_dataset(
    spec: SemanticDatasetSpec,
    *,
    train_transforms: Optional[A.Compose] = None,
    val_transforms: Optional[A.Compose] = None,
    image_size: tuple[int, int] = (512, 512),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    allow_transforms: bool = True,
) -> tuple[DataLoader, Optional[DataLoader]]:
    
    if allow_transforms == True:
        if train_transforms is None:
            train_transforms = TRAIN_TFMS(image_size, mean, std)
        if val_transforms is None:
            val_transforms = VAL_TFMS(image_size, mean, std)
    else:
        train_transforms = None
        val_transforms = None
    
    if not spec.train_set or not spec.val_set:
        raise ValueError("leafseg requires train_files and val_files requires")
    train_pairs = load_split_file_pairs(spec.root, spec.image_dir, spec.mask_dir, spec.train_set)
    val_pairs = load_split_file_pairs(spec.root, spec.image_dir, spec.mask_dir, spec.val_set)

    trn_img, trn_msk = zip(*train_pairs)
    train_ds = PlantDreamerData(trn_img, trn_msk, transforms=train_transforms)

    val_img, val_msk = zip(*val_pairs)
    val_ds = PlantDreamerData(val_img, val_msk, transforms=val_transforms)

    return train_ds, val_ds


# basic smoke test
# if __name__ == "__main__":
#     import argparse

#     logging.basicConfig(level=logging.INFO)

#     ap = argparse.ArgumentParser()
#     ap.add_argument("--registry", type=str, default="datasets.yaml")
#     ap.add_argument("--dataset", type=str, required=True)
#     ap.add_argument("--batch_size", type=int, default=8)
#     ap.add_argument("--num_workers", type=int, default=4)
#     ap.add_argument("--pin_memory", action="store_true")
#     args = ap.parse_args()

#     train_loader, val_loader, spec, split = build_dataloaders(
#         dataset_id=args.dataset,
#         registry_path=args.registry,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_memory,
#     )

#     # iterate one batch to validate
#     x, y = next(iter(train_loader))
#     print("Train batch:", x.shape, y.shape)
#     if val_loader is not None:
#         x, y = next(iter(val_loader))
#         print("Val batch:", x.shape, y.shape)