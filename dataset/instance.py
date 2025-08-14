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




class LeafDataset(Dataset):
    def __init__(self, img_dir, mask_dir, category_json_path, transforms=None, leaf_only=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.imgs = sorted(os.listdir(img_dir))
        self.leaf_only = leaf_only
        
        # Load category mapping from JSON
        with open(category_json_path, "r") as f:
            self.cat_id_to_name = json.load(f)
        
        self.cat_id_to_class_idx = {}
        for k, v in self.cat_id_to_name.items():
            cat_id = int(k)
            if "Leaf" in v:
                self.cat_id_to_class_idx[cat_id] = 1
            elif "Pot" in v and not self.leaf_only:
                self.cat_id_to_class_idx[cat_id] = 2
            elif "Soil" in v and not self.leaf_only:
                self.cat_id_to_class_idx[cat_id] = 3
            elif "Stem" in v and not self.leaf_only:
                self.cat_id_to_class_idx[cat_id] = 4
            else:
                self.cat_id_to_class_idx[cat_id] = 0  # unknown/ignore

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = np.array(Image.open(img_path).convert("RGB"))
        
        mask_path = os.path.join(self.mask_dir, self.imgs[idx].replace(".jpg", ".png"))
        mask = np.array(Image.open(mask_path))
        
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]
        
        masks, labels, boxes = [], [], []
        
        for obj_id in obj_ids:
            label = self.cat_id_to_class_idx.get(obj_id, 0)
            if self.leaf_only and label != 1:
                continue
            if label == 0:
                continue
            
            obj_mask = (mask == obj_id)
            if obj_mask.sum() == 0:
                continue
            
            masks.append(obj_mask)
            labels.append(label)
            
            pos = np.where(obj_mask)
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        if len(masks) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img.shape[0], img.shape[1]), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64) if len(boxes) > 0 else torch.tensor([])
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }
        
        if self.transforms:
            transformed = self.transforms(image=img, masks=[m.numpy() for m in masks])
            img = transformed["image"]
            masks = torch.stack([torch.tensor(m) for m in transformed["masks"]])
            
            boxes = []
            for m in masks:
                pos = torch.where(m)
                if pos[0].numel() == 0:
                    boxes.append([0, 0, 0, 0])
                else:
                    xmin = torch.min(pos[1]).item()
                    xmax = torch.max(pos[1]).item()
                    ymin = torch.min(pos[0]).item()
                    ymax = torch.max(pos[0]).item()
                    boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            target["boxes"] = boxes
            target["masks"] = masks
            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            img = F.to_tensor(img)
        
        return img, target

    def __len__(self):
        return len(self.imgs)




def visualize_sample(img, target, class_idx_to_name=None, alpha=0.4):
    """
    img: tensor [C, H, W] in 0..1
    target: dict from dataset __getitem__
    class_idx_to_name: dict {int: str} mapping class index -> name
    alpha: mask transparency
    """
    # Convert image to numpy for plotting
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    # Create overlay image for masks
    overlay = img_np.copy()

    masks = target["masks"].cpu().numpy() if len(target["masks"]) > 0 else []
    boxes = target["boxes"].cpu().numpy() if len(target["boxes"]) > 0 else []
    labels = target["labels"].cpu().numpy() if len(target["labels"]) > 0 else []

    for i, mask in enumerate(masks):
        
        # Generate a random color
        color = [random.randint(0, 255) for _ in range(3)]
        # Apply mask to overlay
        colored_mask = np.zeros_like(img_np, dtype=np.uint8)
        colored_mask[mask.astype(bool)] = color
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)

        # Draw bounding box
        x_min, y_min, x_max, y_max = boxes[i].astype(int)
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 2)

        # Add label
        if class_idx_to_name:
            class_name = class_idx_to_name.get(labels[i], f"Class {labels[i]}")
        else:
            class_name = f"Class {labels[i]}"
        cv2.putText(overlay, class_name, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Show image
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

# --- Example usage ---
if __name__ == "__main__":
    
    
    category_json_path = "data\\instance\\annotations.json"
    img_dir = "data\\instance\\gt"
    mask_dir = "data\\instance\\mask"

    dataset = LeafDataset(img_dir, mask_dir, category_json_path, transforms=None)
    
    # Build reverse mapping from dataset mapping
    class_idx_to_name = {0: "background", 1: "leaf", 2: "pot", 3: "soil", 4: "stem"}

    # Pick random sample
    idx = random.randint(0, len(dataset) - 1)
    img, target = dataset[idx]

    visualize_sample(img, target, class_idx_to_name, alpha=1)

