import datetime
import time
import logging
import os

import numpy as np
from torch.utils.data import Subset, DataLoader

from leaf_seg.common.build import build_optimiser, build_scheduler
from leaf_seg.common.loss.cedice import CEDiceLoss
from leaf_seg.dataset.plantdreamer_semantic import build_dataloaders, build_dataset
from leaf_seg.semantic.build import build_reporter, setup_model
from leaf_seg.semantic.train import fit, run
from leaf_seg.semantic.finetune import finetune, load_pretrained_weights
from leaf_seg.common.config import SemanticTrainConfig, SemanticFinetuneConfig

from leaf_seg.reporter.semantic import SemanticTrainingReporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET = "bean_semantic_real"

run_start = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
t0 = time.perf_counter()
wall_start = datetime.datetime.now(datetime.timezone.utc)

for x in range(1, 11):
    cfg = SemanticTrainConfig(
        model="segformer",
        encoder="mit_b2",
        dataset=DATASET,
        num_classes=4,
        batch_size=8,
        num_workers=4,
        lr=0.00005,
        epochs=100,
        use_amp=True,
    )
    path = run(cfg=cfg)

elapsed = time.perf_counter() - t0
wall_end = datetime.datetime.now(datetime.timezone.utc)

print(f"Started: {wall_start.isoformat()}")
print(f"Ended:   {wall_end.isoformat()}")
print(f"Elapsed: {elapsed/60:.2f} minutes")