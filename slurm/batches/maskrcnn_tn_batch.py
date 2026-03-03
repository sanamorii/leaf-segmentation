import logging
import datetime
import os
import time

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from leaf_seg.dataset.build import get_dataset_spec
from leaf_seg.dataset.plantdreamer_instance import build_dataset
from leaf_seg.instance.build import build_reporter
from leaf_seg.instance.train import fit, run
from leaf_seg.instance.finetune import finetune, load_pretrained_weights
from leaf_seg.common.config import InstanceTrainConfig, InstanceFinetuneConfig
from leaf_seg.dataset.plantdreamer_instance import coco_collate_fn
from leaf_seg.instance.build import setup_maskrcnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET = "bean_instance_real"
spec = get_dataset_spec(
    dataset_id=DATASET,
    registry_path="data/datasets.yaml",
)
trn_ds, val_ds = build_dataset(spec=spec)

def make_fixed_subset(dataset, fraction=None, n=None, seed=0):
    assert (fraction is None) ^ (n is None), "Specify exactly one of fraction or n"
    rng = np.random.default_rng(seed)

    N = len(dataset)
    k = int(round(N * fraction)) if fraction is not None else int(n)
    k = max(1, min(N, k))

    idx = rng.permutation(N)[:k].tolist()
    return Subset(dataset, idx)

run_start = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
t0 = time.perf_counter()
wall_start = datetime.datetime.now(datetime.timezone.utc)
for x in range(50, 151, 50):
    cfg = InstanceTrainConfig(
        dataset=DATASET,
        num_classes=4,
        batch_size=8,
        num_workers=4,
        lr=0.00005,
        epochs=100,
        use_amp=True,
    )


    trn_sb = make_fixed_subset(trn_ds, fraction=x/10.0, seed=42)
    trn_loader = DataLoader(trn_sb, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=True, drop_last=True, collate_fn=coco_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=False,drop_last=False, collate_fn=coco_collate_fn)

    logger.info("Subset=%s percen=%s", len(trn_sb), x/10.0)

    model = setup_maskrcnn(num_classes=cfg.num_classes, dataset=cfg.dataset, device=cfg.device)
    model.to(cfg.device)
    load_pretrained_weights(model, cfg.ckpt, device=cfg.device, strict_load=cfg.strict_load)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg.output = os.path.join(cfg.output, f"[{run_start}]_sdxl_bean", f"train[subset-{len(trn_sb)}]-{model.name}")
    cfg.num_classes = spec.num_classes
    reporter = build_reporter(cfg=cfg)

    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", factor=0.5, patience=3)

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