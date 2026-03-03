"""Shared evaluation utilities for semantic and instance segmentation."""

import json
import os
import logging
from pathlib import Path

import torch
import torch.nn as nn

from leaf_seg.models.utils import load_ckpt

logger = logging.getLogger(__name__)


def load_semantic_model(model_name: str, encoder: str, checkpoint: str, num_classes: int, device: str) -> nn.Module:
    from leaf_seg.models.modelling import get_smp_model

    ckpt = load_ckpt(checkpoint, map_location=device)
    model = get_smp_model(name=model_name, encoder=encoder, weights=None, classes=num_classes)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


def load_instance_model(checkpoint: str, num_classes: int, device: str) -> nn.Module:
    from leaf_seg.models.maskrcnn_torch import get_model as get_maskrcnn

    model = get_maskrcnn(num_classes=num_classes)
    ckpt = load_ckpt(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def print_report(title: str, metrics: dict, output_dir: str | None = None):
    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("")

    for k, v in metrics.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for sub_k, sub_v in v.items():
                lines.append(f"  {sub_k}: {sub_v:.4f}")
        elif isinstance(v, float):
            lines.append(f"{k}: {v:.4f}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("")

    msg = "\n".join(lines)
    print(msg)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = Path(output_dir) / "results.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(msg)
        logger.info("Saved report to: %s", path)


def save_json_results(metrics: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = Path(output_dir) / "results.json"

    # convert numpy/int types to json-safe
    def _convert(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_convert(metrics), f, indent=2)
    logger.info("Saved JSON results to: %s", path)
