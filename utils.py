import os
import numpy as np
import csv
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import json
import logging


def decode_mask(mask, class_colors):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        color_mask[mask == class_id] = color

    return color_mask


def save_path_pairs_to_csv(pairs, filepath):
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "mask_path"])
        for img, mask in pairs:
            writer.writerow([img, mask])

def create_ckpt(
    cur_itrs: int, model, optimiser, scheduler, tloss, vloss, vscore, epoch: int = None
):
    ckpt = {
        "cur_itrs": cur_itrs,
        "model_state": model.state_dict(),
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
    if epoch is not None:
        ckpt["epoch"] = int(epoch)
    return ckpt


def save_ckpt(checkpoint, path):
    """ save current model with safe directory creation and logging
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(checkpoint, path)
    logging.getLogger(__name__).info("Model saved as %s", path)


def load_ckpt(path, map_location=None):
    """Load a checkpoint file and return the dict. Uses torch.load.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=map_location)
    logging.getLogger(__name__).info("Loaded checkpoint %s", path)
    return ckpt

def save_results():
    return