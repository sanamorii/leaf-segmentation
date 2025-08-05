import os
from glob import glob
import argparse
import numpy as np
import cv2

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
from torchmetrics.segmentation import (
    DiceScore,
    GeneralizedDiceScore,
    MeanIoU,
    HausdorffDistance
)
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as smp_losses
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

from train import train_fn
from dataset.bean import PlantDreamerAllBean, PlantDreamerBean, COLOR_TO_CLASS
from utils import collect_all_data
from loss.cedice import CEDiceLoss
from loss.dice import dice_coeff, multiclass_dice_coeff

def preprocess_image(path, resize=(256,256)):
    image = Image.open(path).convert("RGB")
    trfm = A.Compose([
        A.Resize(resize),
        A.ToTensorV2(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return trfm(image).unsqueeze(0)

def infer_single(model, image: torch.Tensor, device, threshold=0.5):
    model.eval()
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
        pred = (output > threshold).float()

    return pred.cpu().squeeze(0)


def decode_mask(mask, class_colors):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        color_mask[mask == class_id] = color

    return color_mask

def overlay(image, color_mask, alpha=0.5):
    return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

def overlay_mask(image, mask, alpha=0.5):
    # Ensure same spatial dimensions
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Ensure both are 3 channels
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if mask.ndim == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    return cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)

def show_prediction(image_tensor, pred_mask, gt_mask=None):
    fig, axs = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(12, 4))
    axs[0].imshow(image_tensor.squeeze().cpu(), cmap='gray')
    axs[0].set_title("Input Image")

    axs[1].imshow(pred_mask.squeeze().cpu(), cmap='gray')
    axs[1].set_title("Predicted Mask")

    if gt_mask is not None:
        axs[2].imshow(gt_mask.squeeze().cpu(), cmap='gray')
        axs[2].set_title("Ground Truth Mask")

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def get_args():
    return

def main():

    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        encoder_depth=5,
        in_channels=3,
        decoder_attention_type="scse",
        classes=len(COLOR_TO_CLASS),
    )
    ckpt = torch.load("checkpoints/unetplusplus-resnet50_best.pth")
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # infer_single(model, image=preprocess_image("data/real/gt/01.png"))


    return