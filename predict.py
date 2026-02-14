import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import argparse
import cv2

from models import modelling
from models.modelling import ENCODER_CHOICES, MODEL_CHOICES
from dataset.utils import decode_mask, overlay
from dataset.bean import COLOR_TO_CLASS, CLASS_COLORS


def load_image(path, resize=(256, 256)):
    """Load and preprocess a single image."""
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


def save_prediction(save_dir, basename, image, pred_mask, save_overlay=False):
    """Save input image, predicted mask, and optionally overlay."""
    os.makedirs(save_dir, exist_ok=True)

    Image.fromarray(image).save(f"{save_dir}/{basename}_image.png")
    Image.fromarray(pred_mask).save(f"{save_dir}/{basename}_pred.png")

    print(f"Prediction saved to {save_dir}/{basename}_pred.png")

    if save_overlay:
        # Ensure pred_mask is same size as image
        if pred_mask.shape[:2] != image.shape[:2]:
            pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Ensure pred_mask is 3-channel
        if len(pred_mask.shape) == 2:  # (H, W) → (H, W, 3)
            pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

        overlaid = overlay(image, pred_mask)
        Image.fromarray(overlaid).save(f"{save_dir}/{basename}_overlay.png")
        print(f"Overlay saved to {save_dir}/{basename}_overlay.png")


def infer_folder(model, img_dir, save_dir, device, resize=(256, 256), save_overlay=False):
    """Run inference on all images in a folder."""
    img_paths = sorted(
        list(Path(img_dir).glob("*.png")) +
        list(Path(img_dir).glob("*.jpg")) +
        list(Path(img_dir).glob("*.jpeg"))
        )

    model.to(device)
    model.eval()

    with torch.no_grad():
        for img_path in img_paths:
            basename = img_path.stem

            orig_img, img_tensor = load_image(img_path, resize)
            img_tensor = img_tensor.to(device)

            output = model(img_tensor)
            pred = torch.argmax(output, dim=1)  # [B, H, W]
            pred_np = pred.squeeze().cpu().numpy()

            pred_mask_img = decode_mask(pred_np, CLASS_COLORS)

            save_prediction(save_dir, basename, orig_img, pred_mask_img, save_overlay)


def get_model(name, encoder, weights, ckpt_path, num_classes):
    """Load trained model from checkpoint."""
    model = modelling.get_smp_model(name, encoder, weights, num_classes)
    weights_only = False
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=weights_only)
    if weights_only:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model_state'])
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a segmentation model")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES, help="Name of model to use")
    parser.add_argument("--encoder", type=str, required=True, choices=ENCODER_CHOICES, help="Name of model encoder to use")
    parser.add_argument("--weights", type=str, choices=["imagenet"], default=None, help="Pretrained weights for encoder")
    parser.add_argument("--images", type=str, required=True, help="Path to folder containing input images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="predictions", help="Folder to save prediction results")
    parser.add_argument("--resize", type=int, nargs=2, default=[256, 256], help="Resize shape: height width")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    parser.add_argument("--overlay", action="store_true", help="Save overlay of input image and prediction")
    return parser.parse_args()


def main():
    args = parse_args()

    num_classes = len(COLOR_TO_CLASS)
    model = get_model(args.model, args.encoder, args.weights, args.checkpoint, num_classes)

    infer_folder(
        model=model,
        img_dir=args.images,
        save_dir=args.output,
        device=args.device,
        resize=tuple(args.resize),
        save_overlay=args.overlay
    )


if __name__ == "__main__":
    main()
