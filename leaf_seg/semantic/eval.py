"""Semantic segmentation evaluation runner."""

import datetime
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from leaf_seg.common.config import SemanticEvalConfig
from leaf_seg.common.eval import load_semantic_model, print_report, save_json_results
from leaf_seg.common.vis import (
    colorize_mask,
    load_raw_rgb,
    make_grid,
    make_labeled_montage,
    overlay_mask,
    tensor_to_rgb,
    add_title,
)
from leaf_seg.dataset.build import build_dataloaders
from leaf_seg.semantic.metrics import StreamSegMetrics

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: str,
    progress: bool = True,
) -> dict:
    """Run semantic evaluation over a dataloader.

    Returns the full metrics dict from StreamSegMetrics.get_results().
    """
    model.eval()
    metrics = StreamSegMetrics(num_classes)

    it = tqdm(loader, desc="Evaluating") if progress else loader

    for imgs, masks in it:
        imgs = imgs.to(device).float()
        output = model(imgs)

        preds = output.detach().max(dim=1)[1].cpu().numpy()
        targets = masks.detach().cpu().numpy()

        metrics.update(targets, preds)

        if hasattr(it, "set_postfix"):
            r = metrics.get_results()
            it.set_postfix(mIoU=f"{r['mean_iou']:.4f}", mDice=f"{r['mean_dice']:.4f}")

    return metrics.get_results()


@torch.no_grad()
def save_visualisations(
    model: torch.nn.Module,
    dataset,
    num_samples: int,
    device: str,
    output_dir: str | Path,
    dataset_name: str = "",
) -> None:
    """Save prediction montages (Image | GT Mask | Predicted Mask) for a
    subset of the validation set, similar to YOLO's val visualisations.

    Each sample gets a labelled side-by-side montage saved individually,
    and a combined results grid is saved at the end.
    """
    model.eval()
    output_dir = Path(output_dir) / "vis"
    (output_dir / "montage").mkdir(parents=True, exist_ok=True)
    (output_dir / "pred_only").mkdir(parents=True, exist_ok=True)

    # Unwrap Subset to get the base dataset for raw image loading
    base_ds = dataset
    index_map: list[int] | None = None
    if isinstance(base_ds, Subset):
        index_map = list(base_ds.indices)
        base_ds = base_ds.dataset

    rng = np.random.default_rng(42)
    n = min(num_samples, len(dataset))
    idxs = rng.permutation(len(dataset))[:n].tolist()

    montages: list[Image.Image] = []

    for j, ds_idx in enumerate(idxs):
        sample = dataset[ds_idx]
        img_t, gt_mask = sample[0], sample[1] if len(sample) >= 2 else None

        # Forward pass
        logits = model(img_t.unsqueeze(0).to(device))
        pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.int32)

        # Resolve base-dataset index for raw image loading
        raw_idx = index_map[ds_idx] if index_map is not None else ds_idx
        img_rgb = load_raw_rgb(base_ds, raw_idx, out_hw=pred.shape)
        if img_rgb is None:
            img_rgb = tensor_to_rgb(img_t)

        # Prediction overlay
        pred_rgba = colorize_mask(pred)
        pred_overlay = overlay_mask(img_rgb, pred_rgba, alpha=0.45)

        # Save prediction-only overlay
        pred_overlay.save(output_dir / "pred_only" / f"idx{ds_idx:05d}_pred.png")

        base_img = Image.fromarray(img_rgb, mode="RGB")

        # Build montage panels
        panels: list[tuple[str, Image.Image]] = [("Image", base_img)]

        if gt_mask is not None:
            gt_np = gt_mask.detach().cpu().numpy() if torch.is_tensor(gt_mask) else np.asarray(gt_mask)
            if gt_np.ndim == 3 and gt_np.shape[0] == 1:
                gt_np = gt_np[0]
            if gt_np.ndim == 3 and gt_np.shape[-1] == 1:
                gt_np = gt_np[..., 0]
            gt_np = gt_np.astype(np.int32)
            gt_rgba = colorize_mask(gt_np)
            gt_overlay = overlay_mask(img_rgb, gt_rgba, alpha=0.45)
            panels.append(("Ground Truth", gt_overlay))

        panels.append(("Prediction", pred_overlay))

        montage = make_labeled_montage(panels)
        montage.save(output_dir / "montage" / f"idx{ds_idx:05d}_montage.png")
        montages.append(montage)

    # Combined grid
    if montages:
        grid = make_grid(montages, ncols=2, pad=10)
        title = f"Semantic Eval — {dataset_name}" if dataset_name else "Semantic Eval"
        grid = add_title(grid, title)
        grid.save(output_dir / "results.png")
        logger.info("Saved %d visualisation montages to %s", len(montages), output_dir)


def run(cfg: SemanticEvalConfig, registry_path: str = "data/datasets.yaml"):
    """Entry point for ``leaf-seg eval semantic``."""

    _, val_loader, spec = build_dataloaders(
        dataset_id=cfg.dataset,
        registry_path=registry_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.resize,
        shuffle=False,
        drop_last=False,
    )

    model = load_semantic_model(
        model_name=cfg.model,
        encoder=cfg.encoder,
        checkpoint=cfg.checkpoint,
        num_classes=spec.num_classes,
        device=cfg.device,
    )

    results = evaluate(
        model=model,
        loader=val_loader,
        num_classes=cfg.num_classes,
        device=cfg.device,
    )

    title = f"SEMANTIC EVAL — {cfg.model}/{cfg.encoder} on {cfg.dataset}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    cfg.num_classes = spec.num_classes
    cfg.output = os.path.join(cfg.output, f"{timestamp}[eval]-{model.name}")
    print_report(title, results, output_dir=cfg.output)
    save_json_results(results, cfg.output)

    if cfg.save_vis:
        save_visualisations(
            model=model,
            dataset=val_loader.dataset,
            num_samples=cfg.num_vis_samples,
            device=cfg.device,
            output_dir=cfg.output,
            dataset_name=f"{cfg.model}/{cfg.encoder} on {cfg.dataset}",
        )

    return results
