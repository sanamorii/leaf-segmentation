import json
import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt
import random
import cv2

class CVPPPLeafDataset(Dataset):
    """CVPPP dataset"""
    def __init__(self, dir, transforms=None):
        """
        img_dir: folder containing RGB images (ending with _rgb.png)
        mask_dir: folder containing per-leaf label masks (ending with _label.png)
        transforms: optional transformations (Albumentations or torchvision)
        """
        self.dir = dir
        self.transforms = transforms

        # match images and masks by filename prefix
        self.files = sorted([f for f in os.listdir(dir) if f.endswith("_rgb.png")])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_file = self.files[idx]
        prefix = img_file.replace("_rgb.png", "")
        mask_file = prefix + "_label.png"

        # load RGB image
        img = Image.open(os.path.join(self.dir, img_file)).convert("RGB")
        img_np = np.array(img)

        # load label mask
        mask = np.array(Image.open(os.path.join(self.mask_dir, mask_file)))
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]  # skip background

        masks = []
        boxes = []
        labels = []

        for obj_id in obj_ids:
            obj_mask = (mask == obj_id).astype(np.uint8)
            if obj_mask.sum() == 0:
                continue

            # bounding box
            ys, xs = np.where(obj_mask)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            boxes.append([x_min, y_min, x_max, y_max])
            masks.append(obj_mask)
            labels.append(1)  # all leaves -> category 1

        # convert to torch tensors
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, img_np.shape[0], img_np.shape[1]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd
        }

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=img_np, masks=[m.numpy() for m in masks])
            img_np = transformed["image"]
            masks = torch.stack([torch.tensor(m) for m in transformed["masks"]])
            target["masks"] = masks

        # Convert image to tensor
        from torchvision.transforms import ToTensor
        img_tensor = ToTensor()(img_np)

        return img_tensor, target

def visualize_cvppp_sample(img_tensor, target, alpha=0.5):
    """img_tensor: torch.Tensor [C, H, W], values in 0..1"""
    # Convert image to numpy
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    overlay = img_np.copy()

    masks = target["masks"].cpu().numpy() if len(target["masks"]) > 0 else []
    boxes = target["boxes"].cpu().numpy() if len(target["boxes"]) > 0 else []
    labels = target["labels"].cpu().numpy() if len(target["labels"]) > 0 else []

    for i, mask in enumerate(masks):
        color = [random.randint(0, 255) for _ in range(3)]
        colored_mask = np.zeros_like(img_np, dtype=np.uint8)
        colored_mask[mask.astype(bool)] = color
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)

        x_min, y_min, x_max, y_max = boxes[i].astype(int)
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(overlay, f"leaf", (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    dataset = CVPPPLeafDataset("data\\CVPPP\\CVPPP2017_LSC_training\\CVPPP2017_LSC_training\\training\\A1", "data\\CVPPP\\CVPPP2017_LSC_training\\CVPPP2017_LSC_training\\training\\A1")
    img, target = dataset[0]

    visualize_cvppp_sample(img, target)