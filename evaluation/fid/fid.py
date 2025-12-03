import os
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset

from .inception import InceptionV3Features
from .utils import compute_statistics, frechet_distance


class Images(Dataset):
    def __init__(self, folder):
        root = Path(folder) / "gt"
        self.paths = list(root.glob("*"))
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)
    

class MaskedImages(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.img_dir = self.root / "gt"
        self.mask_dir = self.root / "mask"

        assert self.img_dir.is_dir(), f"Missing image folder: {self.img_dir}"
        assert self.mask_dir.is_dir(), f"Missing mask folder: {self.mask_dir}"

        img_paths = self.img_dir.glob("*.png")

        if not img_paths:
            raise RuntimeError(f"No images found in {self.img_dir}")

        # sort to ensure deterministic pairing; assume masks share the same filenames
        self.image_paths = sorted(img_paths)
        self.mask_paths = [self.mask_dir / p.name for p in self.image_paths]

        # sanity check
        missing_masks = [p for p in self.mask_paths if not p.is_file()]
        if missing_masks:
            raise RuntimeError(
                f"Missing masks for {len(missing_masks)} images. "
                f"Example missing mask: {missing_masks[0]}"
            )


        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        img = Image.open(self.image_paths[i]).convert("RGB")
        mask = Image.open(self.mask_paths[i])  # 8-bit categorical mask

        img_np  = np.array(img).astype(np.uint8)
        mask_np = np.array(mask).astype(np.uint8)

        fg = (mask_np != 0).astype(np.uint8) # foreground >= 1, background = 0
        img_np_masked = img_np * fg[..., None]  # set background to zero (black)
        img_masked = Image.fromarray(img_np_masked)

        return self.transform(img_masked)



def get_features(folder, batch_size=32, device="cuda", exclude_bg=False):
    if exclude_bg:
        dataset = MaskedImages(folder)
    else:
        dataset = Images(folder)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = InceptionV3Features().to(device)

    feats = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            f = model(batch)   # [B, 2048]
            feats.append(f.cpu().numpy())

    feats = np.concatenate(feats, axis=0)
    return feats


def compute_fid(real, synth, batch_size=32, device="cuda", exclude_bg=False):

    real_features = get_features(real, batch_size, device, exclude_bg)
    synth_features = get_features(synth, batch_size, device, exclude_bg)

    mu_r, sigma_r = compute_statistics(real_features)
    mu_s, sigma_s = compute_statistics(synth_features)

    fid = frechet_distance(mu_r, sigma_r, mu_s, sigma_s)
    return fid
