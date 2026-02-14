import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from leaf_seg.reporter.semantic import SemanticTrainingReporter
from leaf_seg.semantic.config import SemanticTrainConfig
from models.modelling import get_smp_model as get_model


def setup_model(cfg: SemanticTrainConfig) -> nn.Module:
    model = get_model(
        name=cfg.model,
        encoder=cfg.encoder,
        weights="imagenet", # NOTE: default to imagenet since training from scratch may be long; reconsider later
        classes=cfg.num_classes,
    )

    model.name = f"{model.__class__.__name__.lower()}-{cfg.encoder}-{cfg.dataset}"
    return model

def build_reporter(report_every: int,  model_name: str, cfg: SemanticTrainConfig) -> SemanticTrainingReporter:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{model_name}-{cfg.epochs}-{timestamp}-report"
    report_dir = Path(cfg.out) / "reports" /  run_name
    reporter = SemanticTrainingReporter(
        output_dir=report_dir,
        monitor_metric="mean_iou",  # TODO: set customiseable montior metric
        plot_every=max(1, int(report_every)),
        append=bool(cfg.resume),
    )

    meta = {
        "model": cfg.model,
        "encoder": cfg.encoder,
        "dataset": cfg.dataset,
        "num_classes": cfg.num_classes,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "lr": cfg.lr,
        "epochs": cfg.epochs,
        "device": str(cfg.device),
        "resume": cfg.resume,
        "use_amp": bool(cfg.use_amp),
        "gradient_clipping": cfg.gradient_clipping,
        "montior_metric": cfg.monitor_metric,
        
        # TODO: patience and weights
    }

    reporter.write_metadata(meta)
    return reporter

def build_optimiser(model: nn.Module, lr: float) -> Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

def build_scheduler(optimiser: Optimizer) -> LRScheduler:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3
    )