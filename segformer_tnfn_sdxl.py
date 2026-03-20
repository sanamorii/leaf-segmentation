"""
Grid experiment: vary synthetic training data AND real finetuning data.

For each synthetic subset size, train a SegFormer from scratch on synthetic
data, then for each real-data fraction, finetune the resulting checkpoint
on real data.  This produces a 2-D grid of (synthetic_n, real_frac) results
so you can study the interaction between synthetic pre-training volume and
real finetuning volume.

Outputs are organised as:
    checkpoints/semantic/grid/<run_start>/
        train[synth-<N>]-segformer_mit_b2/          <- training run
        finetune[synth-<N>_real-<M>]-segformer.../   <- finetune run
"""

import datetime
import itertools
import json
import logging
import os
import time

import numpy as np
from torch.utils.data import DataLoader, Subset

from leaf_seg.common.build import build_optimiser, build_scheduler
from leaf_seg.common.config import SemanticFinetuneConfig, SemanticTrainConfig
from leaf_seg.common.loss.cedice import CEDiceLoss
from leaf_seg.dataset.plantdreamer_semantic import build_dataset as build_real_dataset
from leaf_seg.dataset.build import get_dataset_spec
from leaf_seg.dataset.plantdreamer_semantic import build_dataset as build_synth_dataset
from leaf_seg.semantic.build import build_reporter, setup_model
from leaf_seg.semantic.finetune import finetune, load_pretrained_weights
from leaf_seg.semantic.train import fit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fixed_subset(dataset, fraction=None, n=None, seed=0):
    """Deterministic random subset of a dataset."""
    assert (fraction is None) ^ (n is None), "Specify exactly one of fraction or n"
    rng = np.random.default_rng(seed)
    N = len(dataset)
    k = int(round(N * fraction)) if fraction is not None else int(n)
    k = max(1, min(N, k))
    idx = rng.permutation(N)[:k].tolist()
    return Subset(dataset, idx)


# ---------------------------------------------------------------------------
# Configuration — edit these to suit your experiment
# ---------------------------------------------------------------------------

SYNTH_DATASET   = "bean_sdxl_smnt_turntable"   # synthetic dataset id
REAL_DATASET    = "bean_semantic_real"              # real dataset id
REGISTRY        = "data/datasets.yaml"

# Grid axes
SYNTH_SIZES     = list(range(100, 2001, 100))      # absolute counts of synthetic images
REAL_FRACTIONS  = [i / 10.0 for i in range(1, 11)] # 10 % → 100 % of real images

# Shared hyper-parameters
MODEL           = "segformer"
ENCODER         = "mit_b2"
NUM_CLASSES     = 4
BATCH_SIZE      = 8
NUM_WORKERS     = 4
LR              = 5e-5
TRAIN_EPOCHS    = 100       # epochs for synthetic pre-training
FINETUNE_EPOCHS = 100       # epochs for real finetuning
FREEZE_EPOCHS   = 10        # frozen-encoder epochs during finetune
USE_AMP         = True
SEED            = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load datasets once
# ---------------------------------------------------------------------------

synth_spec = get_dataset_spec(dataset_id=SYNTH_DATASET, registry_path=REGISTRY)
synth_trn_ds, synth_val_ds = build_synth_dataset(synth_spec)

real_spec = get_dataset_spec(dataset_id=REAL_DATASET, registry_path=REGISTRY)
real_trn_ds, real_val_ds = build_real_dataset(
    spec=real_spec
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run_start = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
base_output = os.path.join("checkpoints", "semantic", "grid", run_start)
os.makedirs(base_output, exist_ok=True)

results = []          # accumulate per-cell metrics
t0 = time.perf_counter()
wall_start = datetime.datetime.now(datetime.timezone.utc)

for synth_n in SYNTH_SIZES:

    # ----- Phase 1: train from scratch on synthetic data -------------------
    logger.info("===== Training | synth_n=%d =====", synth_n)

    train_cfg = SemanticTrainConfig(
        model=MODEL,
        encoder=ENCODER,
        dataset=SYNTH_DATASET,
        num_classes=synth_spec.num_classes,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        lr=LR,
        epochs=TRAIN_EPOCHS,
        use_amp=USE_AMP,
    )

    synth_subset = make_fixed_subset(synth_trn_ds, n=synth_n, seed=SEED)
    synth_trn_loader = DataLoader(
        synth_subset,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    synth_val_loader = DataLoader(
        synth_val_ds,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    model = setup_model(train_cfg, synth_spec)

    optimiser  = build_optimiser(model, train_cfg.lr)
    scheduler  = build_scheduler(optimiser)
    loss_fn    = CEDiceLoss(ce_weight=0.5, dice_weight=0.5)

    train_output = os.path.join(base_output, f"train[synth-{len(synth_subset)}]-{model.name}")
    train_cfg.output = train_output

    reporter = None
    if not train_cfg.no_report:
        reporter = build_reporter(cfg=train_cfg)

    fit(
        model=model,
        train_loader=synth_trn_loader,
        val_loader=synth_val_loader,
        optimiser=optimiser,
        scheduler=scheduler,
        loss_fn=loss_fn,
        cfg=train_cfg,
        reporter=reporter,
    )

    # Locate the best checkpoint produced by fit()
    ckpt_path = os.path.join(train_output, "model_best.pth")
    if not os.path.isfile(ckpt_path):
        logger.warning("Expected checkpoint not found at %s — skipping finetune for synth_n=%d", ckpt_path, synth_n)
        continue

    # ----- Phase 2: finetune on real data at each fraction -----------------
    for real_frac in REAL_FRACTIONS:

        real_subset = make_fixed_subset(real_trn_ds, fraction=real_frac, seed=SEED)
        logger.info(
            "  Finetuning | synth_n=%d  real_frac=%.0f%%  real_n=%d",
            synth_n, real_frac * 100, len(real_subset),
        )

        ft_cfg = SemanticFinetuneConfig(
            model=MODEL,
            encoder=ENCODER,
            dataset=REAL_DATASET,
            num_classes=real_spec.num_classes,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            lr=LR,
            epochs=FINETUNE_EPOCHS,
            use_amp=USE_AMP,
            ckpt=ckpt_path,
            freeze_epochs=FREEZE_EPOCHS,
        )

        real_trn_loader = DataLoader(
            real_subset,
            batch_size=ft_cfg.batch_size,
            num_workers=ft_cfg.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        real_val_loader = DataLoader(
            real_val_ds,
            batch_size=ft_cfg.batch_size,
            num_workers=ft_cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        ft_model = setup_model(ft_cfg, real_spec)
        load_pretrained_weights(ft_model, ft_cfg.ckpt, device=ft_cfg.device, strict_load=ft_cfg.strict_load)

        ft_loss_fn = CEDiceLoss(ce_weight=0.5, dice_weight=0.5)

        ft_output = os.path.join(
            base_output,
            f"finetune[synth-{len(synth_subset)}_real-{len(real_subset)}]-{ft_model.name}",
        )
        ft_cfg.output = ft_output

        ft_reporter = build_reporter(cfg=ft_cfg)

        best_ckpt = finetune(
            model=ft_model,
            train_loader=real_trn_loader,
            val_loader=real_val_loader,
            loss_fn=ft_loss_fn,
            cfg=ft_cfg,
            reporter=ft_reporter,
        )

        results.append({
            "synth_n":    synth_n,
            "real_frac":  real_frac,
            "real_n":     len(real_subset),
            "train_ckpt": ckpt_path,
            "ft_ckpt":    best_ckpt,
            "ft_output":  ft_output,
        })

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

elapsed  = time.perf_counter() - t0
wall_end = datetime.datetime.now(datetime.timezone.utc)

summary = {
    "started":       wall_start.isoformat(),
    "ended":         wall_end.isoformat(),
    "elapsed_min":   round(elapsed / 60, 2),
    "synth_sizes":   SYNTH_SIZES,
    "real_fractions": REAL_FRACTIONS,
    "runs":          results,
}

summary_path = os.path.join(base_output, "grid_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

logger.info("Grid experiment complete — %d cells", len(results))
logger.info("Summary written to %s", summary_path)

print(f"\nStarted: {wall_start.isoformat()}")
print(f"Ended:   {wall_end.isoformat()}")
print(f"Elapsed: {elapsed / 60:.2f} minutes")
print(f"Total runs: {len(results)}")
print(f"Summary: {summary_path}")