# import numpy as np
# from PIL import Image
# import glob, os

# for folder in glob.glob("./data/beans/*/mask"):
#     print("Folder:", folder)
#     vals = set()
#     for f in glob.glob(os.path.join(folder, "*.png"))[:10]:  # check first 10 masks
#         arr = np.array(Image.open(f))
#         if arr.ndim == 3:  # RGB mask
#             unique_colors = np.unique(arr.reshape(-1, arr.shape[-1]), axis=0)
#             vals.update([tuple(c) for c in unique_colors])
#         else:  # grayscale mask
#             unique_vals = np.unique(arr)
#             vals.update([(v,) for v in unique_vals])  # wrap scalar in tuple for consistency
#     print("Unique RGB values:", sorted(vals))

import random
import matplotlib.pyplot as plt
import os
import numpy as np

from dataset.utils import get_dataloader

def denormalize_image(img_tensor):
    # Convert CHW tensor to HWC numpy array
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    # If normalized with ImageNet mean/std, undo it
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = (img * std + mean)  # undo normalization
    img = np.clip(img, 0, 1)  # clip to valid range
    return img

def test_dataloader(dataset_name, batch_size=4, num_workers=0, augment=False):
    train_loader, val_loader = get_dataloader(dataset_name, 8, num_workers, augment=augment)

    # Pick a random batch from the training loader
    batch = next(iter(train_loader))
    
    # If your dataset returns (image, mask)
    images, masks = batch
    
    # Random index in the batch
    idx = random.randint(0, len(images) - 1)
    img, mask = images[idx], masks[idx]

    # If your dataset keeps file paths for validation, retrieve them
    if hasattr(train_loader.dataset, "pairs"):
        img_path, mask_path = train_loader.dataset.pairs[idx]
        print(f"Image file: {os.path.basename(img_path)}")
        print(f"Mask file:  {os.path.basename(mask_path)}")
        if os.path.basename(img_path) == os.path.basename(mask_path):
            print("✅ Image and mask filenames match")
        else:
            print("❌ Filename mismatch!")

    # Visualize
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(denormalize_image(img))
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    if mask.ndim == 3 and mask.shape[0] == 1:
        plt.imshow(mask.squeeze(0), cmap="gray")
    else:
        plt.imshow(mask)
    plt.title("Mask")
    plt.axis("off")
    plt.savefig("result.png")

test_dataloader("all")