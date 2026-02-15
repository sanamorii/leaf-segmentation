import os
import torch
from glob import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A

CLASS_COLORS = {
    0: (0, 0, 0),       # Background (black)
    1: (0, 255, 0),     # Leaf (green)
    2: (255, 165, 0),   # Pot (orange)
    3: (139, 69, 19),   # Soil (brown)
    4: (157, 0, 255),   # Stem (purple)
}
COLOR_TO_CLASS = {
    (0, 0, 0): 0,  # Background
    (1, 1, 1): 1,  # Leaf
    (2, 2, 2): 2,  # Pot
    (3, 3, 3): 3,  # Soil
    (4, 4, 4): 4   # Stem
}

def rgb_to_class(mask, class_colors):
    mask = np.array(mask)
    # If masks are in 0/255 format, shrink them to 0/1
    if mask.max() > 4:
        mask = mask // 255

    # Map RGB -> class index
    class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for color, class_id in COLOR_TO_CLASS.items():
        match = np.all(mask == color, axis=-1)
        class_mask[match] = class_id

    # Safety check: values must be within 0..N-1
    assert class_mask.max() <= len(COLOR_TO_CLASS)-1, \
        f"Unexpected mask values found. Got max={class_mask.max()}"
    return class_mask

class PlantDreamerAllBean(Dataset):
    def __init__(self, paths, transforms=None, debug=False):
        img, masks = zip(*paths)
        self.image_paths = img
        self.mask_paths = masks
        self.transforms = transforms
        self.debug = debug

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.image_paths[idx]
        mask = self.mask_paths[idx]
        assert os.path.basename(img) == os.path.basename(mask), f"Name mismatch: {img}, {mask}"

        image = np.array(Image.open(img).convert("RGB"))
        mask = np.array(Image.open(mask).convert("RGB"))
        mask = rgb_to_class(mask, class_colors=COLOR_TO_CLASS) 

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image'].float()
            mask = augmented['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        if self.debug:
            uniq = torch.unique(mask)
            if not torch.all((uniq >= 0) & (uniq <= 4)):
                print(f"[WARN] Mask {self.mask_paths[idx]} has unexpected values: {uniq.tolist()}")

        return image, mask


class PlantDreamerBean(Dataset):
    def __init__(self, image_dir:str, mask_dir:str, transforms=None, debug=False) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = list(sorted(os.listdir(image_dir)))
        self.masks = list(sorted(os.listdir(mask_dir)))
        self.transforms = transforms
        self.debug = debug

        super().__init__()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))
        mask = rgb_to_class(mask, class_colors=COLOR_TO_CLASS)  # class mask from RGB

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"].float()
            mask = augmented["mask"].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        if self.debug:
            uniq = torch.unique(mask)
            if not torch.all((uniq >= 0) & (uniq <= 4)):
                print(f"[WARN] Mask {self.mask_paths[index]} has unexpected values: {uniq.tolist()}")

        return image, mask
