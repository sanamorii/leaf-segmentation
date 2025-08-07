import argparse
import numpy as np
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassF1Score,
    MulticlassAccuracy,
)
from torchmetrics.segmentation import (
    DiceScore,
    GeneralizedDiceScore,
    MeanIoU,
    HausdorffDistance
)
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as smp_losses
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from metrics import StreamSegMetrics
from dataset.bean import PlantDreamerAllBean, PlantDreamerBean, COLOR_TO_CLASS, CLASS_COLORS
from dataset.utils import collect_all_data, decode_mask, get_dataloader, overlay

def preprocess_image(path, resize=(256,256)):
    image = Image.open(path).convert("RGB")
    orig = np.array(image)
    trfm = A.Compose([
        A.Resize(resize),
        A.ToTensorV2(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return orig, trfm(image).unsqueeze(0)

def infer(model, image: torch.Tensor, device, threshold=0.5):
    """torch.nograd() and model.eval() need to be active"""
    image = image.to(device)

    output = model(image)
    output = torch.sigmoid(output)
    pred = (output > threshold).float()

    return pred

def save_prediction(path : str, pred):
    pred = pred.max(1)[1].cpu().numpy()[0] #hw
    
    # type of np.ndarray[tuple[int, int, int]]
    colourized_preds = Image.fromarray(decode_mask(mask=pred, class_colors=CLASS_COLORS))
    colourized_preds.save(path)

def save_comparison(path:str, image_tensor, pred_mask, gt_mask=None):
    fig, axs = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(12, 4))
    axs[0].imshow(image_tensor.squeeze().cpu(), cmap='gray')
    axs[0].set_title("Input Image")

    axs[1].imshow(pred_mask.squeeze().cpu(), cmap='gray')
    axs[1].set_title("Predicted Mask")

    if gt_mask is not None:
        axs[2].imshow(gt_mask.squeeze().cpu(), cmap='gray')
        axs[2].set_title("Ground Truth Mask")

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path)


def eval_valset(model, dataloader, num_classes: int, device: str):
    metrics = StreamSegMetrics(num_classes=num_classes)
    meaniou = MeanIoU(num_classes=num_classes, input_format='index').to(device)
    dice = DiceScore(num_classes=num_classes, average='macro', input_format='index').to(device)
    hausdorff = HausdorffDistance(num_classes=num_classes, distance_metric='euclidean', input_format='index').to(device)
    
    model.eval()

    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc="Evaluating"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            meaniou.update(preds, masks)
            dice.update(preds, masks)
            hausdorff.update(preds, masks)
            metrics.update(masks, preds)
    
    meaniou = meaniou.compute().item()
    dice = dice.compute().item()
    hausdorff = hausdorff.compute().item()
    return metrics.get_results() | {'meaniou': meaniou, 'dice': dice, 'hausdorff': hausdorff}


def eval_images(model, dataset: str, num_classes, device, dir):
    metrics = StreamSegMetrics(num_classes=num_classes)
    meaniou = MeanIoU(num_classes=num_classes, input_format='index').to(device)
    dice = DiceScore(num_classes=num_classes, average='macro', input_format='index').to(device)
    hausdorff = HausdorffDistance(num_classes=num_classes, distance_metric='euclidean', input_format='index').to(device)

    data = collect_all_data(dataset)
    model.eval()
    with torch.no_grad():
        for img, mask in data:
            basename = Path(img).stem

            img_orig, img_tensor = preprocess_image(img)
            mask_orig, mask_tensor = preprocess_image(mask)

            img_tensor, mask_tensor = img_tensor.to(device), mask_tensor.to(device)
            output = model(img_tensor)
            pred = torch.argmax(output.squeeze(), dim=0)

            mask_pred = decode_mask(pred.cpu().numpy(), CLASS_COLORS)

            # metrics
            metrics.update([mask_tensor.cpu().numpy()], [pred.cpu().numpy()])
            meaniou.update(pred.unsqueeze(0), mask_tensor.unsqueeze(0))
            dice.update(pred.unsqueeze(0), mask_tensor.unsqueeze(0))
            hausdorff.update(pred.unsqueeze(0), mask_tensor.unsqueeze(0))

            Image.fromarray(img_orig).save(os.path.join(dir, f"{basename}_gt.png"))
            Image.fromarray(mask_orig).save(os.path.join(dir, f"{basename}_mask.png"))
            Image.fromarray(mask_pred).save(os.path.join(dir, f"{basename}_pred.png"))
    results = metrics.get_results() | {'meaniou': meaniou, 'dice': dice, 'hausdorff': hausdorff}
    print(metrics.to_str(results))
    return results

def get_args():
    return

def main():

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        encoder_depth=5,
        in_channels=3,
        decoder_attention_type="scse",
        classes=len(COLOR_TO_CLASS),
    )
    ckpt = torch.load("checkpoints1/unetplusplus-resnet34_best.pth", weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # infer_single(model, image=preprocess_image("data/real/gt/01.png"))

    results = eval_images(model, "eval/images/", 4, "cuda", "results/")

if __name__ == "__main__":
    main()