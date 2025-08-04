import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import 
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from tqdm import tqdm

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as smp_losses
from segmentation_models_pytorch.base.model import SegmentationModel
import matplotlib.pyplot as plt


import os
import sys
import cv2

from dataset.bean import rgb_to_class
from loss.earlystop import EarlyStop, STATUS
from utils import overlay, save_ckpt
from metrics import StreamSegMetrics


def create_checkpoint(cur_itrs: int, model, optimiser: Optimizer, scheduler, tloss, vloss, vscore):
    return {
        "cur_itrs": cur_itrs,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimiser.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "validation_loss": vloss,
        "training_loss": tloss,
        "overall_val_acc": vscore["Overall Acc"],
        "mean_val_acc": vscore["Mean Acc"],
        "freqw_val_acc": vscore["FreqW Acc"],
        "mean_val_iou": vscore["Mean IoU"],
        "class_val_iou": vscore["Class IoU"],
    }


def validate(
    model: SegmentationModel,
    loader: DataLoader,
    metrics: StreamSegMetrics,
    epoch: tuple[int, int],
    loss_fn,
    device: str,
):
    model.eval()
    metrics.reset()
    val_bar = tqdm(loader, desc=f"Epoch {epoch[0]+1}/{epoch[1]} [Val]", leave=False)
    running_vloss = 0

    with torch.no_grad():
        for img, mask in val_bar:
            img = img.to(device).float()
            mask = mask.to(device).long()

            output = model(img)
            # detaches tensor from comp graph, selects the highest score for each pixel
            # cpu() moves from gpu to cpu, numpy() converts from tensor to np array
            preds = output.detach().max(dim=1)[1].cpu().numpy()
            targets = mask.cpu().numpy()

            # update metrics and validation loss
            loss = loss_fn(preds, mask)
            running_vloss += loss.item()
            metrics.update(targets, preds)

        score = metrics.get_results()
    avg_val_loss = running_vloss / len(loader)
    return score, avg_val_loss

def train(model, loader: DataLoader, optimiser, loss_fn, device, epoch: tuple[int, int]):
    running_loss = 0
    last_loss = 0

    for i, (imgs, masks) in enumerate((loader)):
        cur_itrs += 1
        imgs = imgs.to(device, type=torch.float32, non_blocking=True)
        masks = masks.to(device, type=torch.long, non_blocking=True)

        optimiser.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimiser.step()

        np_loss = loss.detach().cpu().numpy()
        running_loss += np_loss


def train_fn(
    model,
    loss_fn,
    optimiser,
    scheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs,
    device,
    num_classes,
    visualise: bool = False,
):

    best_vloss = np.inf
    best_score = 0.0
    cur_itrs = 0
    interval_loss = 0

    stop_policy = EarlyStop(patience=10, min_delta=0.001) # early stopping policy
    metrics = StreamSegMetrics(num_classes)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False
        )

        for imgs, masks in train_bar:
            cur_itrs += 1
            imgs = imgs.to(device, type=torch.float32, non_blocking=True)
            masks = masks.to(device, type=torch.long, non_blocking=True)

            optimiser.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimiser.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_tloss = running_loss / len(train_loader)
        val_score, avg_vloss = validate(
            model=model,
            loader=val_loader,
            metrics=metrics,
            epoch=(epoch, epochs),
            loss_fn=loss_fn,
            device=device,
        )

        scheduler.step(avg_vloss, epoch=epoch)

        print(
            f"Epoch {epoch+1}/{epochs} - Loss, {interval_loss:.4f}, \
                Avg Train Loss: {avg_tloss:.4f}, \
                Avg Val Loss: {avg_vloss:.4f}, \
                "
        )

        # save model
        checkpoint = create_checkpoint(
            cur_itrs=cur_itrs,
            model=model,
            optimiser=optimiser,
            scheduler=scheduler,
            tloss=avg_tloss,
            vloss=avg_vloss,
            vscore=val_score,
        )

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
        if val_score["Mean IoU"] > best_score:
            best_score = val_score["Mean IoU"]
            save_ckpt(checkpoint, f"checkpoints/{model.module.name}_best.pth")
        save_ckpt(checkpoint, f"checkpoints/{model.module.name}_current.pth")

        torch.cuda.empty_cache()  # clear cache

        stop_policy(avg_vloss, model)
        if stop_policy.early_stop:
            print("No improvement in average validation loss - terminating.")
            break
