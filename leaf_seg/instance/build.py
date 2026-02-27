import datetime
from pathlib import Path
from dataclasses import asdict

import torch

from leaf_seg.common.config import InstanceTrainConfig
from leaf_seg.models.maskrcnn_torch import get_model as get_maskrcnn
from leaf_seg.reporter.instance import InstanceTrainingReporter


def setup_maskrcnn(num_classes: int, dataset: str, device: str) -> torch.nn.Module:
    model = get_maskrcnn(num_classes=num_classes)
    model.name = f"maskrcnn-{dataset}"
    return model.to(torch.device(device))


def build_reporter(cfg: InstanceTrainConfig):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = Path(cfg.output)
    reporter = InstanceTrainingReporter(
        output_dir=report_path,
        monitor_metric="segm_AP",
        plot_every=max(1, int(cfg.report_every)),
        append=bool(cfg.resume),
    )

    meta = asdict(cfg)
    meta.update({"started_on": timestamp})

    reporter.write_metadata(meta)
    return reporter