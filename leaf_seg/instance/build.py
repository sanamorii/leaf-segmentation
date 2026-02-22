import datetime
from pathlib import Path

import torch

from leaf_seg.common.config import InstanceTrainConfig
from leaf_seg.models.maskrcnn_torch import get_model as get_maskrcnn
from leaf_seg.reporter.instance import InstanceTrainingReporter


def setup_maskrcnn(num_classes: int, dataset: str, device: str) -> torch.nn.Module:
    model = get_maskrcnn(num_classes=num_classes)
    model.name = f"maskrcnn-{dataset}-instance"
    return model.to(torch.device(device))


def build_reporter(cfg: InstanceTrainConfig):

    report_path = Path(cfg.output)
    reporter = InstanceTrainingReporter(
        output_dir=report_path,
        monitor_metric="segm_AP",
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
    }
    reporter.write_metadata(meta)
    return reporter