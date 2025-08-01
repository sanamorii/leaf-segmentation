import os
from glob import glob

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassAccuracy
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as smp_losses
from sklearn.model_selection import train_test_split

from train import train_fn
from dataset.bean import PlantDreamerAllBean, PlantDreamerBean, COLOR_TO_CLASS
from utils import collect_all_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER = "resnet152"
WEIGHTS = "imagenet"
EPOCHS = 50
BEAN_DATASET=''

def get_dataloader(dataset):
  train_loader = None
  val_loader = None
  if dataset == "bean01":

    train_aug = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ])
    val_aug = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2(),
    ])

    train_ds = PlantDreamerBean(image_dir="../input/syntheticbeans/beans/bean0/gt", mask_dir="../input/syntheticbeans/beans/bean0/mask", transforms=train_aug)
    val_ds = PlantDreamerBean(image_dir="../input/syntheticbeans/beans/bean1/dataset/gt", mask_dir="../input/syntheticbeans/beans/bean1/dataset/mask", transforms=val_aug)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=2)
  if dataset == "all":

    train_aug = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ])
    val_aug = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2(),
    ])

    all_pairs = collect_all_data("../input/syntheticbeans/beans")
    train_paths, val_paths = train_test_split(all_pairs, test_size=0.2, random_state=42)
    train_imgs, train_masks = zip(*train_paths)
    val_imgs, val_masks     = zip(*val_paths)
    train_ds = PlantDreamerAllBean(train_imgs, train_masks, transforms=train_aug)
    val_ds = PlantDreamerAllBean(val_imgs, val_masks, transforms=val_aug)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=True, num_workers=2)
  else:
    raise Exception("invalid dataset")
  return train_loader, val_loader


def main():
    unetplusplus = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=WEIGHTS,
        encoder_depth=5,
        in_channels=3,
        decoder_attention_type='scse',
        classes=len(COLOR_TO_CLASS),
    )
    
    train_loader, val_loader = get_dataloader("all")
    optimiser = torch.optim.Adam(unetplusplus.parameters(), lr=1e-4)
    # loss_fn = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    loss_fn = smp_losses.DiceLoss(mode='multiclass')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=3, factor=0.5)
    train_fn(unetplusplus, loss_fn, optimiser, scheduler, train_loader, val_loader, EPOCHS, 'cuda', visualise=False)
    
if __name__ == "__main__":
    main()