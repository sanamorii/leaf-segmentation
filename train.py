import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
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
import matplotlib.pyplot as plt


import os
import sys
import cv2

from dataset.bean import rgb_to_class
from utils import overlay, save_ckpt

def train_fn(
    model,
    loss_fn,
    optimiser,
    scheduler,
    train_loader,
    val_loader,
    epochs,
    device,
    visualise: bool = False,
):
    model.to(device)
    best_loss = np.inf
    cur_itrs = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False
        )

        for imgs, masks in train_bar:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

            cur_itrs += 1

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)

        with torch.no_grad():
            for imgs, masks in val_bar:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = loss_fn(preds, masks)
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        scheduler.step(avg_val_loss)

        # Save model
        checkpoint = {
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimiser.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_loss,
        }
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_ckpt(checkpoint, f"checkpoints/{model.name}_best.pth")
        save_ckpt(checkpoint, f"checkpoints/{model.name}_best.pth")

        if visualise:
            # Visualize first batch of validation
            imgs, masks = next(iter(val_loader))
            imgs_np = imgs.cpu().permute(0, 2, 3, 1).numpy()
            preds = torch.argmax(model(imgs.to(device)), dim=1).cpu().numpy()
            fig, axes = plt.subplots(
                min(4, len(imgs_np)), 3, figsize=(12, 3 * min(4, len(imgs_np)))
            )
            for i in range(min(4, len(imgs_np))):
                ax = axes[i]
                orig = (imgs_np[i] * 255).astype(np.uint8)
                gt_cm = rgb_to_class(masks[i].numpy())
                pred_cm = rgb_to_class(preds[i])
                ax[0].imshow(orig)
                ax[0].set_title("Image")
                ax[0].axis("off")
                ax[1].imshow(gt_cm)
                ax[1].set_title("GT Mask")
                ax[1].axis("off")
                ax[2].imshow(overlay(orig, pred_cm))
                ax[2].set_title("Overlay")
                ax[2].axis("off")
            plt.tight_layout()
            plt.show()
