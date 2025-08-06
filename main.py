import os
from glob import glob
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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

from loss.poly import PolyLR
from train import train_fn
from dataset.bean import COLOR_TO_CLASS
from utils import collect_all_data
from loss.cedice import CEDiceLoss
from dataset.utils import get_dataloader

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

    parser.add_argument("--shuffle", type=bool, default=True)
    return parser

def get_optimiser(optim, model, lr, **opts) -> torch.optim.Optimizer:
    if optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, **opts)
    elif optim == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, **opts)

def get_policy(policy, optimiser, opts):
    
    if policy == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", patience=3, factor=0.5
        )
    elif policy == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=10000, gamma=0.1
        )
    elif policy == "warmupcosine":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimiser, start_factor=0.1, total_iters=5
        )
        cosine = CosineAnnealingLR(optimiser, T_max=opts.epochs - 5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimiser, schedulers=[warmup, cosine]
        )

    return scheduler

def get_lossfn():
    return

def setup_ddp(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()


def main(rank, world_size):
    opts = get_args().parse_args()
    print(f"-"*50)
    print(f"encoder: {opts.encoder}")
    print(f"weights: {opts.weights}")
    print(f"epochs: {opts.epochs}")

    setup_ddp(rank, world_size)
    device = f"cuda:{rank}"

    model = smp.UnetPlusPlus(
        encoder_name=opts.encoder,
        encoder_weights=opts.weights,
        encoder_depth=5,
        in_channels=3,
        # decoder_channels=[128, 64, 32, 16, 8],  # [256, 128, 64, 32, 16]
        decoder_attention_type="scse",
        classes=len(COLOR_TO_CLASS),
    )
    model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    print("GPUs:", torch.cuda.device_count())
    print("Using", torch.cuda.device_count(), "GPUs")
    print("Model device:", next(model.parameters()).device)
    print("Training model:", model.name)

    train_loader, val_loader = get_dataloader(dataset=opts.dataset, 
                                              batch_size=opts.batch_size,
                                              num_workers=opts.num_workers,
                                              pin_memory=opts.pin_memory,
                                              shuffle=opts.shuffle)
    # optimiser = torch.optim.AdamW(
    #     model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay
    # )

    optimiser = torch.optim.SGD(params=[
        {'params': model.encoder.parameters(), 'lr': 0.1 * opts.learning_rate},
        {'params': model.segmentation_head.parameters(), 'lr': opts.learning_rate},
    ], lr=opts.learning_rate, momentum=0.9, weight_decay=opts.weight_decay)

    loss_fn = CEDiceLoss(ce_weight=0.5, dice_weight=0.5)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=3, factor=0.5
    )

    # scheduler = PolyLR(optimizer=optimiser, max_iters=30e3, power=0.9)

    train_fn(
        model=model,
        loss_fn=loss_fn,
        optimiser=optimiser,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=opts.epochs,
        device=device,
        rank=rank,
        num_classes=len(COLOR_TO_CLASS),
        visualise=False,
    )

    cleanup_ddp()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
