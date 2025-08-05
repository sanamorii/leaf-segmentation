import os
from glob import glob
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import Dataset

from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassF1Score,
    MulticlassAccuracy,
)
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as smp_losses
from sklearn.model_selection import train_test_split

from train import train_fn
from dataset.bean import PlantDreamerAllBean, PlantDreamerBean, COLOR_TO_CLASS
from utils import collect_all_data
from loss.cedice import CEDiceLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", help="training dataset", choices=["all", "bean", "kale"])
    parser.add_argument("--encoder", type=str, default="resnet50", help="", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    parser.add_argument("--weights", type=str, default=None, help="", choices=["imagenet"])

    parser.add_argument("--epochs", type=int, default=50)
    # parser.add_argument("--total_itrs", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--shuffle", type=bool, default=True)
    return parser


def get_dataloader(dataset, batch_size, num_workers, pin_memory=False, shuffle=True):
    if dataset == "bean01":

        train_aug = A.Compose(
            [
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        val_aug = A.Compose(
            [
                A.Resize(256, 256),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

        train_ds = PlantDreamerBean(
            image_dir="./data/beans/bean0/gt",
            mask_dir="./data/beans/bean0/mask",
            transforms=train_aug,
        )
        val_ds = PlantDreamerBean(
            image_dir="./data/beans/bean1/gt",
            mask_dir="./data/beans/bean1/mask",
            transforms=val_aug,
        )
    if dataset == "all":

        train_aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.4),

                # A.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05),
                A.GaussianBlur(p=0.2),
                A.GaussNoise(p=0.3),
                # A.RandomCrop(width=256, height=256, p=1.0), # potentially skipping important features
                A.HueSaturationValue(p=0.4),
                A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        val_aug = A.Compose(
            [
                A.Resize(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        all_pairs = collect_all_data("./data/beans")
        train_paths, val_paths = train_test_split(
            all_pairs, test_size=0.2, random_state=42, shuffle=True
        )
        train_imgs, train_masks = zip(*train_paths)
        val_imgs, val_masks = zip(*val_paths)
        train_ds = PlantDreamerAllBean(train_imgs, train_masks, transforms=train_aug)
        val_ds = PlantDreamerAllBean(val_imgs, val_masks, transforms=val_aug)
    else:
        raise Exception("invalid dataset")
    
    print("Training dataset size: ", len(train_ds))
    print("Validation dataset size: ", len(val_ds))
    print("Dataset type: ", dataset)

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


def get_optimiser(optim, model, lr, **opts) -> torch.optim.Optimizer:
    if optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, **opts)
    elif optim == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, **opts)


def get_lossfn():
    return


def main():
    opts = get_args().parse_args()
    print(f"-"*50)
    print(f"encoder: {opts.encoder}")
    print(f"weights: {opts.weights if opts.weights != None else "transfer learning not used"}")
    print(f"epochs: {opts.epochs}")

    unetplusplus = smp.UnetPlusPlus(
        encoder_name=opts.encoder,
        encoder_weights=opts.weights,
        encoder_depth=5,
        in_channels=3,
        decoder_channels=[128, 64, 32, 16, 8],  # [256, 128, 64, 32, 16]
        decoder_attention_type="scse",
        classes=len(COLOR_TO_CLASS),
    )
    unetplusplus.to(DEVICE)
    model = nn.DataParallel(unetplusplus)  # use multiple gpus

    print("GPUs:", torch.cuda.device_count())
    print("Using", torch.cuda.device_count(), "GPUs")
    print("Model device:", next(model.parameters()).device)
    print("Training model:", model.module.name)

    train_loader, val_loader = get_dataloader(dataset=opts.dataset, 
                                              batch_size=opts.batch_size,
                                              num_workers=opts.num_workers,
                                              pin_memory=opts.pin_memory,
                                              shuffle=opts.shuffle)
    # optimiser = torch.optim.AdamW(
    #     model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay
    # )

    optimiser = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.learning_rate},
        {'params': model.classifier.parameters(), 'lr': opts.learning_rate},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    loss_fn = CEDiceLoss(ce_weight=0.5, dice_weight=0.5)
    
    # if opts.policy == "plateau":
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimiser, mode="min", patience=3, factor=0.5
    #     )
    # elif opts.policy == "step":
    #     scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimiser, step_size=10000, gamma=0.1
    #     )
    # elif opts.policy == "warmupcosine":
    #     warmup = torch.optim.lr_scheduler.LinearLR(
    #         optimiser, start_factor=0.1, total_iters=5
    #     )
    #     cosine = CosineAnnealingLR(optimiser, T_max=opts.epochs - 5)
    #     scheduler = torch.optim.lr_scheduler.SequentialLR(
    #         optimiser, schedulers=[warmup, cosine]
    #     )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=3, factor=0.5
    )

    train_fn(
        model=model,
        loss_fn=loss_fn,
        optimiser=optimiser,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=opts.epochs,
        device=DEVICE,
        num_classes=len(COLOR_TO_CLASS),
        visualise=False,
    )


if __name__ == "__main__":
    main()
