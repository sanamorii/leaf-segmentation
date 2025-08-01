
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


def collect_all_data(base_dir):
    pairs = []

    for subdir in os.listdir(base_dir):
        gt_dir = os.path.join(base_dir, subdir, "gt")
        mask_dir = os.path.join(base_dir, subdir, "mask")

        if not os.path.exists(gt_dir) or not os.path.exists(mask_dir):
            continue

        images = sorted(glob(os.path.join(gt_dir, "*.png")))
        masks = sorted(glob(os.path.join(mask_dir, "*.png")))

        # assuming 1-to-1 filename match
        assert len(images) == len(masks)
        for img, mask in zip(images, masks):
            assert os.path.basename(img) == os.path.basename(mask)
        pairs += zip(images, masks)
            
    return pairs

def decode_mask(mask, class_colors):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        color_mask[mask == class_id] = color

    return color_mask

def overlay(image, color_mask, alpha=0.5):
    return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)


def preprocess_image(image_path, device, resize=(256, 256)):
    image = Image.open(image_path).convert("RGB")
    orig = np.array(image)
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    return orig, tensor

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