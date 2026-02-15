import os
import torch
from glob import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


CLASS_COLORS = {
    0: (0, 0, 0),       # Background (black)
    1: (0, 255, 0),     # Leaf (green)
    2: (255, 165, 0),   # Pot (orange)
    3: (139, 69, 19),   # Stem (brown)
    # 4: (157, 0, 255),   # Stem (purple)
}
COLOR_TO_CLASS = {
    (0, 0, 0): 0,  # Background
    (1, 1, 1): 1,  # Leaf
    (2, 2, 2): 2,  # Pot
    (3, 3, 3): 3,  # Soil
    # (4, 4, 4): 4   # Stem
}

def rgb_to_class(mask, class_colors):
    mask = np.array(mask)
    class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for color, class_id in class_colors.items():
        match = np.all(mask == color, axis=-1)
        class_mask[match] = class_id
    return class_mask

class PlantDreamerAllKale(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("RGB"))
        mask = rgb_to_class(mask, class_colors=COLOR_TO_CLASS) 

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image'].float()
            mask = augmented['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask


class PlantDreamerAllKale(Dataset):
    def __init__(self, image_dir:str, mask_dir:str, transforms=None) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.images = list(sorted(os.listdir(image_dir)))
        self.masks = list(sorted(os.listdir(mask_dir)))

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
        return image, mask
