
import os
import numpy as np
import csv
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import json


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
    cur_itrs: int, model, optimiser, scheduler, tloss, vloss, vscore
):
    return {
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


def save_ckpt(checkpoint, path):
    """ save current model
    """
    torch.save(checkpoint, path)
    print("Model saved as %s" % path)

def save_results():
    return