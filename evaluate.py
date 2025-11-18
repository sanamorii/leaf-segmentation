import os
import numpy as np
import torch
import argparse

from tqdm import tqdm
import segmentation_models_pytorch as smp
import torchvision.transforms as T
import albumentations as A

from pathlib import Path
from PIL import Image
from torchmetrics.segmentation import MeanIoU, DiceScore, HausdorffDistance

from metrics import StreamSegMetrics
from models import modelling
from models.modelling import ENCODER_CHOICES, MODEL_CHOICES
from dataset.utils import decode_mask, overlay
from dataset.plantdreamer_semantic import COLOR_TO_CLASS, CLASS_COLORS

def load_image(path, resize=(256, 256)):
    image = Image.open(path).convert("RGB")
    orig = np.array(image)
    transform = T.Compose([
        T.Resize(resize),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return orig, tensor

def load_mask(path : str, n_classes : int, resize=(256, 256)):
    mask = Image.open(path)
    mask = np.array(mask)

    transform = A.Compose([
        A.Resize(height=resize[0], width=resize[1])
    ])
    transformed = transform(image=mask)
    mask = transformed["image"]

    if len(mask.shape) > 2:
        raise RuntimeError("Mask is not monochrome, aborting")
    if len(np.unique(mask)) > n_classes:
        raise RuntimeError(f"Number of classes in mask is not equal to reported no. of classes.\nViolating mask: {path}")

    mask = np.clip(mask, 0, max(n_classes))
    tensor = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

    return mask, tensor

def save_visuals(save_dir, basename, image, pred_mask, gt_mask=None):
    os.makedirs(save_dir, exist_ok=True)

    Image.fromarray(image).save(f"{save_dir}/{basename}_image.png")
    Image.fromarray(pred_mask).save(f"{save_dir}/{basename}_pred.png")
    if gt_mask is not None:
        Image.fromarray(gt_mask).save(f"{save_dir}/{basename}_gt.png")

def evaluate_folder(model, img_dir, mask_dir, save_dir, num_classes, device, verbosity, resize=(256, 256)):
    img_paths = sorted(Path(img_dir).glob("*.png"))
    mask_paths = sorted(Path(mask_dir).glob("*.png"))

    assert len(img_paths) == len(mask_paths), "Mismatch between images and masks"

    model.to(device)
    model.eval()

    metrics = StreamSegMetrics(num_classes)

    if verbosity > 1:
        loader = tqdm(zip(img_paths, mask_paths), total=len(img_paths), desc="Evaluating")
    else:
        loader = zip(img_paths, mask_paths)

    with torch.no_grad():
        for img_path, mask_path in loader:
            basename = img_path.stem

            orig_img, img_tensor = load_image(img_path, resize)
            gt_mask_img, gt_tensor = load_mask(mask_path, resize)

            img_tensor, gt_tensor = img_tensor.to(device), gt_tensor.to(device)

            output = model(img_tensor)
            pred = torch.argmax(output, dim=1)  # [B, H, W]
            pred_np = pred.squeeze().cpu().numpy()
            gt_np = gt_tensor.squeeze().cpu().numpy()

            pred_mask_img = decode_mask(pred_np, CLASS_COLORS)

            # Update metrics
            metrics.update(gt_np, pred_np)

            # Save prediction, GT, input
            save_visuals(save_dir, basename, orig_img, pred_mask_img, gt_mask_img)
    results = metrics.get_results()
    print("Evaluation Results:")
    print(f"Mean IoU           : {results['Mean IoU']:.4f}")
    print(f"Dice Coefficient   : {results['Mean Dice']:.4f}")
    print(f"Accuracy   : {results['Mean Acc']:.4f}")
    print(metrics.to_str(results))
    print("\n")



def get_model(ckpt_path, num_classes):
    weights_only = False
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=weights_only)

    # [0] - name, [1] - encoder, [2] - dataset
    name, encoder, *_ = checkpoint["model_name"].split("-")
    model = modelling.get_model(name=name, encoder=encoder, weights=None, num_classes=num_classes)

    if weights_only:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model_state'])
    return model


def get_model_manual():
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model on image folder")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES, help="Name of model to use")
    parser.add_argument("--encoder", type=str, required=True, choices=ENCODER_CHOICES, help="Name of model encoder to use")
    parser.add_argument("--weights", type=str, choices=["imagenet"], default=None, help="Name of pretrained weights to use for transfer learning")
    parser.add_argument("--images", type=str, required=True, help="Path to folder containing input images")
    parser.add_argument("--masks", type=str, required=True, help="Path to folder containing ground truth masks")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="results", help="Folder to save prediction results")
    parser.add_argument("--resize", type=int, nargs=2, default=[256, 256], help="Resize shape: height width")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    parser.add_argument("--verbosity", type=int, default=2, help="Verbosity level")
    return parser.parse_args()


def main():
    args = parse_args()

    num_classes = len(COLOR_TO_CLASS)
    model = get_model(args.model, args.encoder, args.weights, args.checkpoint, num_classes)
    if not os.path.isdir(args.masks): raise Exception(f"not valid folder: {args.masks}")
    if not os.path.isdir(args.images): raise Exception(f"not valid folder: {args.images}")

    if args.verbosity > 0:
        print("-"*50)
        print(f"Encoder: {args.encoder}")
        print(f"Weights: {args.weights}")
        print(f"Model: {model.name}")
        print(f"Using: {args.checkpoint}")
        print("-"*50)

    evaluate_folder(
        model=model,
        img_dir=args.images,
        mask_dir=args.masks,
        save_dir=args.output,
        num_classes=num_classes,
        device=args.device,
        resize=tuple(args.resize),
        verbosity=args.verbosity,
    )


if __name__ == "__main__":
    main()
