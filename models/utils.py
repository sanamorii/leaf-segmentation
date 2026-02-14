import os
from pathlib import Path
from typing import Any, Dict
import logging

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

import random
import numpy as np

from segmentation_models_pytorch.base.model import SegmentationModel

def _get_rng_state() -> Dict[str, Any]:
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state['torch_cuda_all'] = torch.cuda.get_rng_state_all()
    return state

def _set_rng_state(state: Dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])

def create_ckpt(
    cur_itrs: int, 
    model : nn.Module, 
    num_classes : int,
    optimiser, 
    scheduler, 
    train_stats: Dict[Any, Any],
    val_stats: Dict[Any, Any],
    epoch: int = None
):
    ckpt = {
        "cur_itrs": cur_itrs,
        "model_name": model.name,
        "model_state": model.state_dict(),
        "optimizer_state": optimiser.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "training_stats": train_stats,
        "val_stats": val_stats,
        "num_classes": num_classes
    }
    if epoch is not None:
        ckpt["epoch"] = int(epoch)
    return ckpt


def save_ckpt(checkpoint, path):
    """ save current model with safe directory creation and logging
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(checkpoint, path)
    logging.getLogger(__name__).info("Model saved as %s", path)


def load_ckpt(path, map_location=None):
    """Load a checkpoint file and return the dict. Uses torch.load.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=map_location)
    logging.getLogger(__name__).info("Loaded checkpoint %s", path)
    return ckpt


def create_maskrcnn_ckpt(
        model: nn.Module,
        optimiser: Optimizer,
        scheduler: LRScheduler | None,
        scaler: GradScaler | None,
        epoch: int,
        train_stats: dict,
        val_stats: dict,
):
    return {
        "epoch": epoch,
        "model_name": f"maskrcnn",
        "model_state": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "train_stats": train_stats,
        "val_stats": val_stats, 
    }
