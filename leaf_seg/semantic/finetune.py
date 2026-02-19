import datetime
import logging
import os

import torch
import torch.nn as nn

from leaf_seg.common.loss.cedice import CEDiceLoss
from leaf_seg.common.verbose import resolve_progress_flag
from leaf_seg.dataset.plantdreamer_semantic import get_dataloader
from leaf_seg.semantic.config import SemanticFinetuneConfig
from leaf_seg.semantic.build import build_reporter, build_scheduler, setup_model
from leaf_seg.models.utils import load_ckpt
from leaf_seg.semantic.train import fit

logger = logging.getLogger(__name__)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in ckpt:
                return ckpt[key]
    return ckpt


def load_pretrained_weights(model, ckpt_path, device, strict_load=False):
    ckpt = load_ckpt(ckpt_path, map_location=device)
    state_dict = _extract_state_dict(ckpt)

    if strict_load:
        model.load_state_dict(state_dict, strict=True)
        logger.info("Loaded pretrained weights (strict) from %s", ckpt_path)
        return

    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered[k] = v
        else:
            skipped.append(k)

    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
    logger.info(
        "Loaded %d/%d tensors from %s (skipped %d mismatched)",
        len(filtered),
        len(state_dict),
        ckpt_path,
        len(skipped),
    )
    if missing_keys:
        logger.info("Missing keys after load: %s", missing_keys)
    if unexpected_keys:
        logger.info("Unexpected keys after load: %s", unexpected_keys)


def freeze_encoder(model):
    if hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False
        logger.info("Encoder frozen")
    else:
        logger.warning("Model has no encoder - freezing unsuccessful")


def unfreeze_encoder(model):
    if hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = True
        logger.info("Encoder unfrozen")
    else:
        logger.warning("Model has no encoder - unfreeze unsuccessful")


def build_finetune_optimiser(model, encoder_lr, decoder_lr, weight_decay):
    enc_params = []
    dec_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder" in name:
            enc_params.append(p)
        else:
            dec_params.append(p)

    param_groups = []
    if enc_params:
        param_groups.append({"params": enc_params, "lr": encoder_lr})
    if dec_params:
        param_groups.append({"params": dec_params, "lr": decoder_lr})

    if not param_groups:
        raise RuntimeError("No trainable parameters found for finetuning.")
    return torch.optim.AdamW(param_groups, lr=decoder_lr, weight_decay=weight_decay)


def run(cfg: SemanticFinetuneConfig):
    device = torch.device(cfg.device)

    train_loader, val_loader = get_dataloader(
        cfg.dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        num_classes=cfg.num_classes,
    )

    model = setup_model(cfg)
    model.to(device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    cfg.output = os.path.join(cfg.output, f"{timestamp}-finetune-{model.name}")
    cfg.progress = resolve_progress_flag(cfg.progress)

    reporter = None
    if not cfg.no_report:
        reporter = build_reporter(cfg=cfg, model_name = model.name)


    # load pretrained weights (strict or shape-matched)
    load_pretrained_weights(model, cfg.ckpt, device=cfg.device, strict_load=cfg.strict_load)

    loss_fn = CEDiceLoss(ce_weight=0.5, dice_weight=0.5)
    encoder_lr = cfg.encoder_lr if cfg.encoder_lr is not None else cfg.lr * 0.1
    decoder_lr = cfg.decoder_lr if cfg.decoder_lr is not None else cfg.lr

    # stage 1: optional frozen encoder
    freeze_epochs = cfg.freeze_epochs
    freeze_encoder_flag = cfg.freeze_encoder
    remaining_epochs = cfg.epochs
    if freeze_epochs > 0:
        freeze_epochs = min(freeze_epochs, cfg.epochs)
        freeze_encoder_flag = True

        if freeze_encoder_flag:
            freeze_encoder(model)

        optimiser = build_finetune_optimiser(
            model=model,
            encoder_lr=encoder_lr,
            decoder_lr=decoder_lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = build_scheduler(optimiser)

        fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimiser=optimiser,
            scheduler=scheduler,
            loss_fn=loss_fn,
            cfg=cfg,
            reporter=reporter,
            start_epoch=0,
            end_epoch=freeze_epochs
        )
        remaining_epochs = cfg.epochs - freeze_epochs
        if remaining_epochs <= 0:
            return

        unfreeze_encoder(model)

    # stage 2 (or single stage): full finetuning
    if freeze_encoder_flag and freeze_epochs == 0:
        freeze_encoder(model)

    optimiser = build_finetune_optimiser(
        model=model,
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_scheduler(optimiser)

    start_epoch = cfg.epochs - remaining_epochs
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimiser=optimiser,
        scheduler=scheduler,
        loss_fn=loss_fn,
        cfg=cfg,
        reporter=reporter,
        start_epoch=start_epoch,
        end_epoch=cfg.epochs
    )
