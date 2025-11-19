import os
import numpy as np
import torch
import click

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

def save_result(path, array):
    Image.fromarray(array).save(path)

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

def infer_single(model, img_path, save_dir, device, resize=(256,256)):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        orig, tensor = load_image(img_path, resize)
        tensor = tensor.to(device)

        output = model(tensor)
        pred = torch.argmax(output, dim=1)  # [B, H, W]
        pred_np = pred.squeeze().cpu().numpy()
    pred_mask_img = decode_mask(pred_np, CLASS_COLORS)

    stem = Path(img_path).stem
    save_result(f"{save_dir}/{stem}_orig.png", orig)
    save_result(f"{save_dir}/{stem}_pred.png", pred_mask_img)

def load_model(model_name, encoder, ckpt_path, num_classes, weights=None):
    weights_only = False
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=weights_only)

    model = modelling.get_model(name=model_name, encoder=encoder, weights=weights, classes=num_classes)

    if weights_only:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model_state'])
    return model


def get_model_manual():
    pass

def verbose_header(verbosity, model_name, encoder, weights, ckpt):
    if verbosity > 0:
        click.echo("-" * 50)
        click.echo(f"Encoder: {encoder}")
        click.echo(f"Weights: {weights}")
        click.echo(f"Model:   {model_name}")
        click.echo(f"Using checkpoint: {ckpt}")
        click.echo("-" * 50)


@click.group()
def cli():
    """Segmentation model inference or evaluation"""
    pass


@cli.command()
@click.option("--model", required=True)
@click.option("--encoder", required=True)
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--num_classes", required=True, type=int)
@click.option("--image", type=click.Path(exists=True))
@click.option("--images", type=click.Path(exists=True))
@click.option("--output", default="result", type=str)
@click.option("--device", default="cuda")
@click.option("--resize", default=(256,256), nargs=2, type=int)
@click.option("--verbosity", "-v", count=True, help="Increase verbosity (-v, -vv, -vvv).")
def infer(model, encoder, checkpoint, num_classes, image, images, output, device, resize, verbosity):
    model = load_model(model, encoder, checkpoint, num_classes)
    model.to(device)

    verbose_header(verbosity, model, encoder, None, checkpoint)

    if image:
        infer_single(model, str(image), output, device, resize)
        click.echo(f"Saved inference results for {image} -> {output}")
        return
    
    if images:
        for img_path in Path(images).glob("*.png"):
            infer_single(model, str(img_path), output, device, resize)
        click.echo(f"Saved inference results for folder {images} -> {output}")
        return

    raise click.UsageError("Provide either --image or --images")

@cli.command()
@click.option("--model", required=True)
@click.option("--encoder", required=True)
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--images", required=True, type=click.Path(exists=True))
@click.option("--masks", required=True, type=click.Path(exists=True))
@click.option("--output", default="eval_results")
@click.option("--device", default="cuda")
@click.option("--resize", default=(256,256), nargs=2, type=int)
@click.option("--verbosity", "-v", count=True, help="Increase verbosity.")
def evaluate(model, encoder, checkpoint, images, masks, output, device, resize, verbosity):
    if not os.path.isdir(masks): raise Exception(f"not valid folder: {masks}")
    if not os.path.isdir(images): raise Exception(f"not valid folder: {images}")

    num_classes = len(COLOR_TO_CLASS)
    model = load_model(model, encoder, checkpoint, num_classes)
    model.to(device)

    verbose_header(verbosity, model, encoder, None, checkpoint)

    evaluate_folder(
        model,
        images,
        masks,
        output,
        num_classes,
        device,
        resize,
        verbosity,
    )

    click.echo("Evaluation complete.")

if __name__ == "__main__":
    cli()
