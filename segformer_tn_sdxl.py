import datetime
import time
import logging
import os

import numpy as np
from torch.utils.data import Subset, DataLoader

from leaf_seg.common.config import SemanticTrainConfig
from leaf_seg.common.build import build_optimiser, build_scheduler
from leaf_seg.common.loss.cedice import CEDiceLoss
from leaf_seg.dataset.plantdreamer_semantic import build_dataset
from leaf_seg.dataset.build import get_dataset_spec
from leaf_seg.semantic.build import build_reporter, setup_model
from leaf_seg.semantic.train import fit


def make_fixed_subset(dataset, fraction=None, n=None, seed=0):
    assert (fraction is None) ^ (n is None), "Specify exactly one of fraction or n"
    rng = np.random.default_rng(seed)

    N = len(dataset)
    k = int(round(N * fraction)) if fraction is not None else int(n)
    k = max(1, min(N, k))

    idx = rng.permutation(N)[:k].tolist()
    return Subset(dataset, idx)



logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET = "bean_sdxl_semantic_turntable"
spec = get_dataset_spec(
    dataset_id=DATASET,
    registry_path="data/datasets.yaml",
)
trn_ds, val_ds = build_dataset(spec)

run_start = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
t0 = time.perf_counter()
wall_start = datetime.datetime.now(datetime.timezone.utc)

for x in range(100, 2001, 100):
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

    trn_sb = make_fixed_subset(trn_ds, n=x, seed=42)
    trn_loader = DataLoader(trn_sb, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=True, drop_last=True,)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=False,drop_last=False,)

    
    cfg.num_classes = spec.num_classes
    model = setup_model(cfg, spec)
    
    optimiser = build_optimiser(model, cfg.lr)
    scheduler = build_scheduler(optimiser)
    loss_fn = CEDiceLoss(ce_weight=0.5, dice_weight=0.5)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    cfg.output = os.path.join(cfg.output, f"[{run_start}]_segformer_trains", f"train[subset-{len(trn_sb)}]-{model.name}")

    reporter = None
    if not cfg.no_report:
        reporter = build_reporter(cfg=cfg)

    fit(
        model=model,
        train_loader=trn_loader,
        val_loader=val_loader,
        optimiser=optimiser,
        scheduler=scheduler,
        loss_fn=loss_fn,
        cfg=cfg,
        reporter=reporter,
    )

elapsed = time.perf_counter() - t0
wall_end = datetime.datetime.now(datetime.timezone.utc)

print(f"Started: {wall_start.isoformat()}")
print(f"Ended:   {wall_end.isoformat()}")
print(f"Elapsed: {elapsed/60:.2f} minutes")