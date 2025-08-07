
import os
import numpy as np
import cv2
from PIL import Image
import csv
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

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
            
def save_ckpt(checkpoint, path):
    """ save current model
    """
    torch.save(checkpoint, path)
    print("Model saved as %s" % path)

