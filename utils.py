
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

def plot_data(img,mask,class_colours,save=False):

  # Load image and predicted mask
  image = np.array(Image.open(img).convert("RGB"))
  mask = np.array(Image.open(mask))  # shape [H, W] with class IDs

  # Convert mask to color
  color_mask = decode_mask(mask, class_colours)

  # Overlay
  overlayed = overlay(image, color_mask, alpha=0.5)

  # Show
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 3, 1)
  plt.title("Original")
  plt.imshow(image)
  plt.axis("off")

  plt.subplot(1, 3, 2)
  plt.title("Segmentation Mask")
  plt.imshow(color_mask)
  plt.axis("off")

  plt.subplot(1, 3, 3)
  plt.title("Overlayed")
  plt.imshow(overlayed)
  plt.axis("off")
  plt.tight_layout()
  if save:
    plt.savefig(f"./results/{Path(img).stem}_img.png")
  else:
    plt.show()