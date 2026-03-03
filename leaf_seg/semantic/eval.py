"""Semantic segmentation evaluation runner."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from leaf_seg.common.config import SemanticEvalConfig
from leaf_seg.common.eval import load_semantic_model, print_report, save_json_results
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


def run(cfg: SemanticEvalConfig, registry_path: str = "data/datasets.yaml"):
    """Entry point for ``leaf-seg eval semantic``."""

    model = load_semantic_model(
        model_name=cfg.model,
        encoder=cfg.encoder,
        checkpoint=cfg.checkpoint,
        num_classes=cfg.num_classes,
        device=cfg.device,
    )

    _, val_loader, spec = build_dataloaders(
        dataset_id=cfg.dataset,
        registry_path=registry_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.resize,
        shuffle=False,
        drop_last=False,
    )

    results = evaluate(
        model=model,
        loader=val_loader,
        num_classes=cfg.num_classes,
        device=cfg.device,
    )

    title = f"SEMANTIC EVAL — {cfg.model}/{cfg.encoder} on {cfg.dataset}"
    print_report(title, results, output_dir=cfg.output)
    save_json_results(results, cfg.output)

    return results
