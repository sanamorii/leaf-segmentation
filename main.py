import argparse
import datetime

import optuna
import torch
import torch.optim as optim
import numpy as np

import segmentation_models_pytorch as smp

from models.modelling import get_model
from loss.earlystop import EarlyStopping
from metrics import StreamSegMetrics
from models.unetdropout import UNETDropout
from train import create_ckpt, train_epoch, train_loop, validate_epoch
from dataset.bean import COLOR_TO_CLASS
from loss.cedice import CEDiceLoss
from dataset.utils import get_dataloader
from utils import save_ckpt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CHOICES = [
    "segformer",
    "unet",
    "unetplusplus",
    "unetdropout",
    "fpn",
    "deeplabv3plus",
    "deeplabv3",
]
ENCODER_CHOICES = [
    "mit_b0",
    "mit_b1",
    "mit_b2",
    "mit_b3",
    "mit_b4",
    "mit_b5",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",
    "efficientnet-b5",
    "efficientnet-b6",
    "efficientnet-b7",
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="model to use", choices=MODEL_CHOICES
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="training dataset",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet50",
        help="",
        choices=ENCODER_CHOICES,
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="", choices=["imagenet"]
    )

    parser.add_argument("--optimiser", type=str, default="rmsprop")
    parser.add_argument("--policy", type=str, default="plateau")

    parser.add_argument("--epochs", type=int, default=50)
    # parser.add_argument("--total_itrs", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--gradient_clip", type=float, default=1.0)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--shuffle", type=bool, default=True)

    parser.add_argument("--patience", type=int, default=10)

    return parser

def get_policy(policy, optimiser, opts):
    if policy == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", patience=10, factor=0.1118140163664775
        )
    elif policy == "step":
        return torch.optim.lr_scheduler.StepLR(optimiser, step_size=10000, gamma=0.1)
    elif policy == "warmupcosine":
        warmup_epochs = max(1, int(0.05 * opts.epochs))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimiser, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=opts.epochs - warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimiser,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    elif policy == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=opts.epochs, eta_min=1e-4
        )
    elif policy == "poly":
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer=optimiser, total_iters=30e3, power=0.9
        )
    elif policy == "exponentiallr":
        return torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.8333338826004195)
    else:
        raise Exception("invalid policy")


def get_optimiser(name, model, opts):
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay
        )
    elif name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay,
            betas=(0.847624784863984, 0.9417463044535068),
            eps=1.9789670128284147e-07
        )
    elif name == "nadam":
        return torch.optim.NAdam(
            model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay,
            betas=(0.9739694517764637, 0.9152662833350875),
            eps=3.1561571626342603e-10,
            momentum_decay=0.05009315053929311,
        )
    elif name == "sgd":
        return torch.optim.SGD(
            params=[
                {"params": model.encoder.parameters(), "lr": 0.1 * opts.learning_rate},
                {
                    "params": model.segmentation_head.parameters(),
                    "lr": opts.learning_rate,
                },
            ],
            lr=opts.learning_rate,
            momentum=0.9,
            weight_decay=opts.weight_decay,
        )
    elif name == "rmsprop":
        return optim.RMSprop(
            model.parameters(),
            lr=opts.learning_rate,
            weight_decay=opts.weight_decay,
            momentum=0.999,
            foreach=True,
        )
    else:
        raise Exception(f"Invalid optimiser: {name}")

def get_lossfn():
    return

def main():
    opts = get_args().parse_args()

    model = get_model(opts.model, opts.encoder, opts.weights, classes=len(COLOR_TO_CLASS))
    model.to(DEVICE)

    print("GPUs:", torch.cuda.device_count())
    print("Using", torch.cuda.device_count(), "GPUs")
    print("Model device:", next(model.parameters()).device)

    print("-" * 50)

    print(f"Encoder: {opts.encoder}")
    print(f"Weights: {opts.weights}")
    print(f"Epochs: {opts.epochs}")
    print("Training model:", model.name)
    print(
        f"Optimiser (lr {opts.learning_rate}, wd {opts.weight_decay}): {opts.optimiser}"
    )
    print("Learning Rate Policy: ", opts.policy)
    print("-" * 50)
    print("Training on:", opts.dataset)

    train_loader, val_loader = get_dataloader(
        dataset=opts.dataset,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        pin_memory=opts.pin_memory,
        shuffle=opts.shuffle,
        augment=False
    )

    loss_fn = CEDiceLoss(ce_weight=0.5, dice_weight=0.5)
    optimiser = get_optimiser(name=opts.optimiser, opts=opts, model=model)
    scheduler = get_policy(policy=opts.policy, optimiser=optimiser, opts=opts)

    train_loop(
        model=model,
        loss_fn=loss_fn,
        optimiser=optimiser,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=opts.epochs,
        patience=opts.patience,
        device=DEVICE,
        num_classes=len(COLOR_TO_CLASS),
        visualise=False,
        use_amp=True,
        ckpt_prefix=f"[{opts.optimiser}.{opts.policy}]_{opts.dataset}"
    )


if __name__ == "__main__":
    main()
