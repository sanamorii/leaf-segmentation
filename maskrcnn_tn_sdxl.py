import datetime
import time
import logging
import os

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from leaf_seg.common.config import InstanceTrainConfig
from leaf_seg.dataset.plantdreamer_instance import build_dataset, coco_collate_fn
from leaf_seg.dataset.build import get_dataset_spec
from leaf_seg.instance.build import build_reporter, setup_maskrcnn
from leaf_seg.instance.train import fit


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

DATASET = "bean_sdxl_inst"
spec = get_dataset_spec(
    dataset_id=DATASET,
    registry_path="data/datasets.yaml",
)
trn_ds, val_ds = build_dataset(spec)

run_start = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
t0 = time.perf_counter()
wall_start = datetime.datetime.now(datetime.timezone.utc)

for x in range(100, 1201, 100):
    cfg = InstanceTrainConfig(
        dataset=DATASET,
        num_classes=4,
        batch_size=4,
        num_workers=4,
        lr=0.00005,
        epochs=100,
        use_amp=True,
        progress=False
    )

    trn_sb = make_fixed_subset(trn_ds, n=x, seed=42)
    trn_loader = DataLoader(trn_sb, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=True, drop_last=True,collate_fn=coco_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=False,drop_last=False,collate_fn=coco_collate_fn)
    
    # trn_loader = DataLoader(trn_sb, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=True, drop_last=True, collate_fn=lambda batch: tuple(zip(*batch)))
    # val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=False, drop_last=False, collate_fn=lambda batch: tuple(zip(*batch)))

    
    cfg.num_classes = spec.num_classes
    model = setup_maskrcnn(num_classes=spec.num_classes, dataset=cfg.dataset, device=cfg.device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=3
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    cfg.output = os.path.join(cfg.output, f"[{run_start}]_maskrcnn_trains", f"train[subset-{len(trn_sb)}]-{model.name}")

    reporter = None
    if not cfg.no_report:
        reporter = build_reporter(cfg=cfg)

    fit(
        model=model,
        train_loader=trn_loader,
        val_loader=val_loader,
        optimiser=optimiser,
        scheduler=scheduler,
        cfg=cfg,
        reporter=reporter,
    )

elapsed = time.perf_counter() - t0
wall_end = datetime.datetime.now(datetime.timezone.utc)

print(f"Started: {wall_start.isoformat()}")
print(f"Ended:   {wall_end.isoformat()}")
print(f"Elapsed: {elapsed/60:.2f} minutes")