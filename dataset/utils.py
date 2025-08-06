import os
import cv2
from glob import glob
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .bean import PlantDreamerAllBean, PlantDreamerBean
from sklearn.model_selection import train_test_split

TRAIN_AUG = A.Compose(
    [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        # A.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05),
        A.GaussianBlur(p=0.2),
        A.GaussNoise(p=0.3),
        # A.RandomCrop(width=256, height=256, p=1.0), # potentially skipping important features
        A.HueSaturationValue(p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ]
)
VAL_AUG = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ]
)

PREPROCESS = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ]
)

# train_aug = A.Compose(
#     [
#         A.Resize(256, 256),
#         A.HorizontalFlip(p=0.5),
#         A.RandomBrightnessContrast(p=0.2),
#         A.Normalize(),
#         A.ToTensorV2(),
#     ]
# )
# val_aug = A.Compose(
#     [
#         A.Resize(256, 256),
#         A.Normalize(),
#         A.ToTensorV2(),
#     ]
# )


def collect_all_data(base_dir, ext="png") -> tuple[str, str]:
    pairs = []

    for subdir in os.listdir(base_dir):
        gt_dir = os.path.join(base_dir, subdir, "gt")
        mask_dir = os.path.join(base_dir, subdir, "mask")

        if not os.path.exists(gt_dir) or not os.path.exists(mask_dir):
            continue

        images = sorted(glob(os.path.join(gt_dir, f"*.{ext}")))
        masks = sorted(glob(os.path.join(mask_dir, f"*.{ext}")))

        # assuming 1-to-1 filename match
        assert len(images) == len(masks)
        for img, mask in zip(images, masks):
            assert os.path.basename(img) == os.path.basename(mask)
        pairs += zip(images, masks)

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
    dataset, rank, world_size, batch_size, num_workers, pin_memory=False, shuffle=True, transforms=False,
):
    if dataset == "bean01":
        train_ds = PlantDreamerBean(
            image_dir="./data/beans/bean0/gt",
            mask_dir="./data/beans/bean0/mask",
            transforms=TRAIN_AUG,
        )
        val_ds = PlantDreamerBean(
            image_dir="./data/beans/bean1/gt",
            mask_dir="./data/beans/bean1/mask",
            transforms=VAL_AUG,
        )
    if dataset == "all":
        all_pairs = collect_all_data("./data/beans")
        train_paths, val_paths = train_test_split(
            all_pairs, test_size=0.2, random_state=42, shuffle=shuffle
        )
        train_imgs, train_masks = zip(*train_paths)
        val_imgs, val_masks = zip(*val_paths)
        train_ds = PlantDreamerAllBean(train_imgs, train_masks, transforms=TRAIN_AUG if transforms is True else PREPROCESS)
        val_ds = PlantDreamerAllBean(val_imgs, val_masks, transforms=VAL_AUG  if transforms is True else PREPROCESS)
    else:
        raise Exception("invalid dataset")

    print("Training dataset size: ", len(train_ds))
    print("Validation dataset size: ", len(val_ds))
    print("Dataset type: ", dataset)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank)


    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=val_sampler,
    )
    return train_loader, val_loader
