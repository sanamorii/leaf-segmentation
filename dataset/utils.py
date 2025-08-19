
import os
import cv2
from glob import glob
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .bean import PlantDreamerAllBean, PlantDreamerBean
from sklearn.model_selection import train_test_split


VAL_AUG = A.Compose([
    A.Resize(256, 256), 
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)), 
    A.ToTensorV2()])
    
TRAIN_AUG = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
    A.ToTensorV2()])


def collect_all_data(base_dir, ext="png") -> list[tuple[str, str]]:
    pairs = []
    for subdir in os.listdir(base_dir):
        gt_dir = os.path.join(base_dir, subdir, "gt")
        mask_dir = os.path.join(base_dir, subdir, "mask")

        if not os.path.exists(gt_dir) or not os.path.exists(mask_dir):
            continue

        images = sorted(glob(os.path.join(gt_dir, f"*.{ext}")))
        masks = sorted(glob(os.path.join(mask_dir, f"*.{ext}")))

        assert len(images) == len(masks), f"Image/mask count mismatch in {subdir}"
        for img, mask in zip(images, masks):
            assert os.path.basename(img) == os.path.basename(mask), f"Name mismatch: {img}, {mask}"
        pairs.extend(zip(images, masks))

    return pairs


def decode_mask(mask, class_colors):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        color_mask[mask == class_id] = color

    return color_mask


def overlay(image, color_mask, alpha=0.5):
    return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)


def get_dataloader(dataset, batch_size, num_workers, pin_memory=False, shuffle=True, augment=True):

    if dataset.lower() == "bean":
        train_ds = PlantDreamerBean(
            image_dir="./data/bean/bean0/gt",
            mask_dir="./data/bean/bean0/mask",
            transforms=TRAIN_AUG if augment else VAL_AUG,
        )
        val_ds = PlantDreamerBean(
            image_dir="./data/bean_improved/bean1/gt",
            mask_dir="./data/bean_improved/bean1/mask",
            transforms=VAL_AUG,
        )
    if dataset.lower() == "bean_improved":
        train_ds = PlantDreamerBean(
            image_dir="./data/bean_improved/bean1/gt",
            mask_dir="./data/bean_improved/bean1/mask",
            transforms=TRAIN_AUG if augment else VAL_AUG,
        )
        val_ds = PlantDreamerBean(
            image_dir="./data/bean_improved/bean2/gt",
            mask_dir="./data/bean_improved/bean2/mask",
            transforms=VAL_AUG,
        )
    if dataset.lower() == "bean_real":
        train_ds = PlantDreamerBean(
            image_dir="./data/val_real/bean0/gt",
            mask_dir="./data/val_real/bean0/mask",
            transforms=TRAIN_AUG if augment else VAL_AUG,
        )
        val_ds = PlantDreamerBean(
            image_dir="./data/test_real/gt",
            mask_dir="./data/test_real/mask",
            transforms=VAL_AUG,
        )
    elif dataset.lower() == "all":
        paths = collect_all_data("./data/beans")
        train_pairs, val_pairs = train_test_split(paths, train_size=0.8, random_state=42)
        train_ds = PlantDreamerAllBean(train_pairs, transforms=TRAIN_AUG if augment else VAL_AUG)
        val_ds = PlantDreamerAllBean(val_pairs, transforms=VAL_AUG)
    elif dataset.lower() == "improved":
        paths = collect_all_data("./data/bean_improved")
        train_pairs, val_pairs = train_test_split(paths, train_size=0.8, random_state=42)
        train_ds = PlantDreamerAllBean(train_pairs, transforms=TRAIN_AUG if augment else VAL_AUG)
        val_ds = PlantDreamerAllBean(val_pairs, transforms=VAL_AUG)
    elif dataset.lower() == "mixed":
        paths = collect_all_data("./data/bean_mix")
        train_pairs, val_pairs = train_test_split(paths, train_size=0.8, random_state=42, shuffle=shuffle)
        train_ds = PlantDreamerAllBean(train_pairs, transforms=TRAIN_AUG if augment else VAL_AUG)
        val_ds = PlantDreamerAllBean(val_pairs, transforms=VAL_AUG)
    elif dataset.lower() == "mixedall":
        paths = collect_all_data("./data/bean_mixall")
        train_pairs, val_pairs = train_test_split(paths, train_size=0.8, random_state=42, shuffle=shuffle)
        train_ds = PlantDreamerAllBean(train_pairs, transforms=TRAIN_AUG if augment else VAL_AUG)
        val_ds = PlantDreamerAllBean(val_pairs, transforms=VAL_AUG)
    else:
        raise Exception(f"invalid dataset {dataset}")


    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
