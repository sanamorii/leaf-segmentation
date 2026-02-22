import datetime
from pathlib import Path

import torch.nn as nn

from leaf_seg.reporter.semantic import SemanticTrainingReporter
from leaf_seg.common.config import SemanticTrainConfig
from leaf_seg.models.modelling import get_smp_model as get_model


def setup_model(cfg: SemanticTrainConfig) -> nn.Module:
    model = get_model(
        name=cfg.model,
        encoder=cfg.encoder,
        weights="imagenet", # NOTE: default to imagenet since training from scratch may be long; reconsider later
        classes=cfg.num_classes,
    )

    model.name = f"{model.__class__.__name__.lower()}_{cfg.encoder}_{cfg.dataset}"
    return model

def build_reporter(cfg: SemanticTrainConfig) -> SemanticTrainingReporter:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # run_name = f"{model_name}-{cfg.epochs}-{timestamp}-report"
    report_dir = Path(cfg.output)
    reporter = SemanticTrainingReporter(
        output_dir=report_dir,
        monitor_metric=cfg.metric_to_track,
        plot_every=max(1, int(cfg.report_every)),
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
        "montior_metric": cfg.metric_to_track,
        
        # TODO: patience and weights
    }

    reporter.write_metadata(meta)
    return reporter

