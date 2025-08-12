import datetime
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
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
from loss.earlystop import EarlyStopping
from utils import create_ckpt, save_ckpt
from metrics import StreamSegMetrics




def validate_epoch(
    model: SegmentationModel,
    loader: DataLoader,
    metrics: StreamSegMetrics,
    loss_fn,
    device: str,
    epochs: tuple[int, int],
):
    model.eval()
    metrics.reset()
    val_bar = tqdm(loader, desc=f"Epoch {epochs[0]+1}/{epochs[1]} [Val]", leave=False)
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
            loss = loss_fn(output, mask)
            running_vloss += loss.item()
            metrics.update(targets, preds)
            val_bar.set_postfix(loss=loss.item())

        score = metrics.get_results()
    
    elapsed_time = val_bar.format_dict['elapsed']
    avg_val_loss = running_vloss / len(loader)
    return elapsed_time, score, avg_val_loss


def train_epoch(
    model: SegmentationModel,
    loss_fn,
    optimiser: Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    loader: DataLoader,
    device: str,
    epochs: tuple[int, int],
    use_amp: bool = False,           # mixed precision vs default precision (FP16)
    gradient_clipping: float = 1.0,  # prevent exploding gradients
):
    model.train()
    running_loss = 0

    train_bar = tqdm(
        loader, desc=f"Epoch {epochs[0]+1}/{epochs[1]} [Train]", leave=False
    )

    for imgs, masks in train_bar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)


        optimiser.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type=device, enabled=True):
                outputs = model(imgs)
                loss = loss_fn(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            scaler.step(optimiser)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            
            loss.backward()
            optimiser.step()

        loss_item = loss.item()
        running_loss += loss_item
        train_bar.set_postfix(loss=loss_item)
    elapsed_time = train_bar.format_dict['elapsed']

    avg_loss = running_loss / len(loader)
    return elapsed_time, avg_loss

def train_fn(
    model : SegmentationModel,
    loss_fn,
    optimiser,
    scheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs : int,
    device,
    num_classes : int,
    use_amp: bool = False,
    gradient_clipping: float = 0.1,
    visualise: bool = False,
):

    best_vloss = np.inf
    best_score = 0.0
    cur_itrs = 0

    grad_scaler = torch.amp.GradScaler(device, enabled=use_amp)
    loss_stop_policy = EarlyStopping(patience=10, delta=0.001)  # early stopping policy
    metrics = StreamSegMetrics(num_classes)

    for epoch in range(epochs):

        elapsed_ttime, avg_tloss = train_epoch(
            model=model,
            loss_fn=loss_fn,
            optimiser=optimiser,
            scaler=grad_scaler,
            loader=train_loader,
            device=device,
            epochs=(epoch, epochs),
            gradient_clipping=gradient_clipping,
            use_amp=use_amp
        )

        cur_itrs += len(train_loader)

        elapsed_vtime, val_score, avg_vloss = validate_epoch(
            model=model,
            loader=val_loader,
            metrics=metrics,
            epochs=(epoch, epochs),
            loss_fn=loss_fn,
            device=device,
        )

        scheduler.step(avg_vloss)  # epoch param is deprecated

        print(
            f"Epoch {epoch+1}/{epochs} - Avg Train Loss: {avg_tloss:.4f}, Avg Val Loss: {avg_vloss:.4f}, Mean IoU: {val_score['Mean IoU']:.4f}"
        )
        print(f"Training time: {str(datetime.timedelta(seconds=int(elapsed_ttime)))}, ", end="")
        print(f"Validation time: {str(datetime.timedelta(seconds=int(elapsed_vtime)))}")

        # save model
        checkpoint = create_ckpt(
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
            save_ckpt(checkpoint, f"checkpoints/{model.name}_{epochs}_best.pth")
        save_ckpt(checkpoint, f"checkpoints/{model.name}_{epochs}_current.pth")

        torch.cuda.empty_cache()  # clear cache

        loss_stop_policy(val_score["Mean IoU"])
        if loss_stop_policy.early_stop:
            print("No improvement in mean IoU - terminating.")
            break
