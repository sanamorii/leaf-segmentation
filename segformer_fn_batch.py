import datetime
import time
import logging
import os

import numpy as np
from torch.utils.data import Subset, DataLoader

from leaf_seg.common.build import build_optimiser, build_scheduler
from leaf_seg.common.loss.cedice import CEDiceLoss
from leaf_seg.dataset.build import build_dataloaders, build_dataset
from leaf_seg.semantic.build import build_reporter, setup_model
from leaf_seg.semantic.train import fit, run
from leaf_seg.semantic.finetune import finetune, load_pretrained_weights
from leaf_seg.common.config import SemanticTrainConfig, SemanticFinetuneConfig

from leaf_seg.reporter.semantic import SemanticTrainingReporter

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

DATASET = "bean_semantic_real"
trn_ds, val_ds = build_dataset(
    dataset_id=DATASET,
    registry_path="data/datasets.yaml",
)


run_start = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
t0 = time.perf_counter()
wall_start = datetime.datetime.now(datetime.timezone.utc)

for x in range(1, 11):
    cfg = SemanticFinetuneConfig(
        model="segformer",
        encoder="mit_b2",
        dataset=DATASET,
        num_classes=4,
        batch_size=8,
        num_workers=4,
        lr=0.00005,
        epochs=100,
        use_amp=True,
        ckpt="checkpoints/semantic/train/20260220225611-train-segformer_mit_b2_bean_semantic_synth/model_best.pth",
        freeze_epochs=10,
    )


    trn_sb = make_fixed_subset(trn_ds, fraction=x/10.0, seed=42)
    trn_loader = DataLoader(trn_sb, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=True, drop_last=True,)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, shuffle=False,drop_last=False,)

    logger.info("Subset=%s percen=%s", len(trn_sb), x/10.0)

    model = setup_model(cfg)
    load_pretrained_weights(model, cfg.ckpt, device=cfg.device, strict_load=cfg.strict_load)
    
    loss_fn = CEDiceLoss(ce_weight=0.5, dice_weight=0.5)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    cfg.output = os.path.join(cfg.output, f"[{run_start}]_segformer_finetunes", f"finetune[subset-{len(trn_sb)}]-{model.name}")
    reporter = build_reporter(cfg=cfg)
    path = finetune(
        model=model,
        train_loader=trn_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        cfg=cfg,
        reporter=reporter,
    )

elapsed = time.perf_counter() - t0
wall_end = datetime.datetime.now(datetime.timezone.utc)

print(f"Started: {wall_start.isoformat()}")
print(f"Ended:   {wall_end.isoformat()}")
print(f"Elapsed: {elapsed/60:.2f} minutes")