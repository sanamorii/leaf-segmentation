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

def load_mask(path : str, n_classes : int, resize : tuple=(256, 256)):
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
        raise RuntimeError(f"Number of classes in mask is not equal to reported no. of classes {np.unique(mask)}, {n_classes}.\nViolating mask: {path}")

    mask = np.clip(mask, 0, n_classes)
    tensor = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

    return mask, tensor

def save_visuals(save_dir, basename, image, pred_mask, gt_mask=None):
    os.makedirs(f"{save_dir}/predictions", exist_ok=True)

    Image.fromarray(image).save(f"{save_dir}/predictions/{basename}_image.png")
    Image.fromarray(pred_mask).save(f"{save_dir}/predictions/{basename}_pred.png")
    if gt_mask is not None:
        Image.fromarray(gt_mask).save(f"{save_dir}/predictions/{basename}_gt.png")

def save_result(path, array):
    Image.fromarray(array).save(path)

def display_report(ckpt: dict, metrics: StreamSegMetrics, output: str = None):
    results = metrics.get_results()

    lines = []
    lines.append(f"RESULTS FOR")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Overall Accuracy : {results['Overall Acc']:.4f}")
    lines.append(f"Mean Accuracy    : {results['Mean Acc']:.4f}")
    lines.append(f"Mean IoU         : {results['Mean IoU']:.4f}")
    lines.append(f"Mean Dice        : {results['Mean Dice']:.4f}")
    lines.append(f"FreqW Acc        : {results['FreqW Acc']:.4f}")
    lines.append("")

    lines.append("Per-class IoU:")
    for cls, val in results["Class IoU"].items():
        lines.append(f"  Class {cls}: {val:.4f}")
    lines.append("")

    lines.append("Per-class Dice:")
    for cls, val in results["Class Dice"].items():
        lines.append(f"  Class {cls}: {val:.4f}")
    lines.append("")

    msg = "\n".join(lines)

    print(msg)

    if output is not None:
        os.makedirs("results", exist_ok=True)
        path = Path(output) / "results.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(msg)
        print(f"Saved report to: {path}")

def evaluate_folder(model, img_dir : str, mask_dir : str, save_dir : str, num_classes : int , device : str, verbosity : int, resize=(256, 256)):
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
            gt_mask_img, gt_tensor = load_mask(mask_path, num_classes, resize)

            img_tensor, gt_tensor = img_tensor.to(device), gt_tensor.to(device)

            output = model(img_tensor)
            pred = torch.argmax(output, dim=1)  # [B, H, W]
            pred_np = pred.squeeze().cpu().numpy()
            gt_np = gt_tensor.squeeze().cpu().numpy()

            pred_mask_img = decode_mask(pred_np, CLASS_COLORS)
            gt_mask_img = decode_mask(gt_np, CLASS_COLORS)

            # Update metrics
            metrics.update(gt_np, pred_np)

            # Save prediction, GT, input
            save_visuals(save_dir, basename, orig_img, pred_mask_img, gt_mask_img)
    return metrics


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
@click.option("--num_classes", required=True, type=int)
@click.option("--images", required=True, type=click.Path(exists=True))
@click.option("--masks", required=True, type=click.Path(exists=True))
@click.option("--output", default="results")
@click.option("--device", default="cuda")
@click.option("--resize", default=(256,256), nargs=2, type=int)
@click.option("--verbosity", "-v", count=True, help="Increase verbosity.")
def evaluate(model, encoder, checkpoint, num_classes, images, masks, output, device, resize, verbosity):
    if not os.path.isdir(masks): raise Exception(f"not valid folder: {masks}")
    if not os.path.isdir(images): raise Exception(f"not valid folder: {images}")

    model = load_model(model, encoder, checkpoint, num_classes)
    model.to(device)
    print(verbosity)
    verbose_header(verbosity, model, encoder, None, checkpoint)

    metrics = evaluate_folder(
        model=model,
        img_dir=images,
        mask_dir=masks,
        save_dir=output,
        num_classes=num_classes,
        device=device,
        resize=resize,
        verbosity=verbosity,
    )

    display_report(checkpoint, metrics, output=output)

    click.echo("Evaluation complete.")

if __name__ == "__main__":
    cli()
