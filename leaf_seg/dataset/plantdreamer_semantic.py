from dataclasses import dataclass
import yaml
import json
import torch
import numpy as np
import albumentations as A
import logging

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from typing import List, Literal, Tuple, Optional
from pathlib import Path
from PIL import Image
from glob import glob


logger = logging.getLogger(__name__)

SplitKind = Literal["ratio", "files", "none"]

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


@dataclass(frozen=True, kw_only=True)
class DatasetSpec:
    name: str
    root: Path
    task: Literal["semantic", "instance"]
    image_dir: str = "images"
    mask_dir: str = "masks"
    ext: str = "png"
    # manifest  ## e.g. root/manifest.jsonl

@dataclass(frozen=True, kw_only=True)
class SplitSpec:
    kind: SplitKind
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    train_files: Optional[Path] = None
    val_files: Optional[Path] = None


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

def resolve_pairs(spec: DatasetSpec, split: SplitSpec, shuffle: bool = True) -> tuple[list[tuple[str,str]], list[tuple[str,str]] | None]:

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


def load_registry(registry_path: str | Path) -> dict:
    registry_path = Path(registry_path)

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry YAML not found: {registry_path}")
    
    with registry_path.open(mode="r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)

    if not isinstance(reg, dict):
        raise ValueError("Registry YAML must be a mapping (dict) at the top level")

    return reg

def get_dataset_spec(dataset_id: str, registry_path: str | Path) -> DatasetSpec:
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
    
    return DatasetSpec(
        name=dataset_id,
        root=Path(root),
        task=task,
        image_dir=cfg.get("image_dir", "images"),
        mask_dir=cfg.get("mask_dir", "masks"),
        ext=cfg.get("ext", "png"),
    )


def _resolve_path(base_dir: Path, value: str | Path | None) -> Optional[Path]:
    if value is None or base_dir is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p)

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
) -> tuple[DataLoader, Optional[DataLoader], DatasetSpec, SplitSpec]:
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

    val_ds: DataLoader | None = None
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
) -> tuple[DataLoader, Optional[DataLoader], DatasetSpec, SplitSpec]:

    train_ds, val_ds, spec, split = build_dataset(
        dataset_id=dataset_id,
        registry_path=registry_path,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        image_size=image_size,
        mean=mean, std=std,
        shuffle=shuffle,
        drop_last=drop_last,
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