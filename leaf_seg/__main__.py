# leaf_seg/cli.py
from __future__ import annotations

import logging
import click


from leaf_seg.semantic.train import run as semantic_train_run
from leaf_seg.semantic.finetune import run as semantic_finetune_run
from leaf_seg.semantic.eval import run as semantic_eval_run
# from leaf_seg.semantic.infer import run as semantic_infer_run
from leaf_seg.common.config import (
    SemanticTrainConfig,
    SemanticInferConfig,
    SemanticEvalConfig,
    SemanticFinetuneConfig,
    InstanceTrainConfig,
    InstanceFinetuneConfig,
    InstanceEvalConfig,
)

from leaf_seg.instance.train import run as instance_train_run
from leaf_seg.instance.finetune import run as instance_finetune_run
from leaf_seg.instance.eval import run as instance_eval_run
# from leaf_seg.instance.infer import run as instance_infer_run

logger = logging.getLogger(__name__)


# shared options
def apply_opts(*decorators):
    """Apply click decorators in a readable order."""
    def wrap(f):
        for d in reversed(decorators):
            f = d(f)
        return f
    return wrap


def opt_progress():
    return click.option(
        "--progress/--no-progress",
        default=None,
        help="Enable/disable tqdm progress bars (default: auto based on TTY).",
    )


def opt_device(default="cuda"):
    return click.option("--device", default=default, show_default=True)

def opt_output(default: str):
    return click.option("-o", "--output", "output", type=click.Path(), default=default, show_default=True)


def opt_resize(default=None):
    # default can be (512,512) or None
    if default is None:
        return click.option("--resize", default=None, nargs=2, type=int, help="Optional H W resize.")
    return click.option("--resize", default=default, nargs=2, type=int, show_default=True)


# Root CLI
@click.group(context_settings={"show_default": True})
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging verbosity.",
)
@click.option("--seed", type=int, default=0, help="Global RNG seed.")
@click.pass_context
def cli(ctx: click.Context, log_level: str, seed: int):
    """leaf-segmentation (semantic + instance)"""
    logging.basicConfig(level=getattr(logging, log_level.upper()), format="%(asctime)s %(levelname)s: %(message)s")
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level.upper()
    ctx.obj["seed"] = seed


@cli.group()
def train():
    """Training commands."""
    pass


@cli.group()
def finetune():
    """Fine-tuning commands."""
    pass


@cli.group()
def eval():
    """Evaluation commands."""
    pass


@cli.group()
def infer():
    """Inference commands."""
    pass


# =============================================================================
# TRAIN
# =============================================================================

# train semantic
@train.command("semantic")
@apply_opts(
    opt_device(),
    opt_progress(),
    opt_output("checkpoints/semantic/train"),
)
@click.option("--model", type=str, required=True, help="Segmentation model name (e.g., unet, deeplabv3+).")
@click.option("--encoder", type=str, required=True, help="Encoder backbone name (e.g., resnet34).")
@click.option("--dataset", type=str, required=True, help="Dataset key/name (passed into get_dataloader).")
@click.option("--num-classes", type=int, required=True, help="Number of semantic classes.")
@click.option("--batch-size", type=int, default=8)
@click.option("--num-workers", type=int, default=4)
@click.option("--lr", type=float, default=1e-3)
@click.option("--epochs", type=int, default=100)
@click.option("--resume", type=click.Path(exists=True), default=None)
@click.option("--use-amp", is_flag=True)
@click.option("--gradient-clipping", type=float, default=0.1)
@click.option("--no-report", is_flag=True, help="Disable reporter output.")
@click.option("--report-every", type=int, default=1)
def train_semantic(**kwargs):
    """
    leaf-seg train semantic ...
    """
    cfg = SemanticTrainConfig(**kwargs)
    semantic_train_run(cfg)


# train instance
@train.command("instance")
@apply_opts(
    opt_device(),
    opt_progress(),
    opt_output("checkpoints/instance/train"),
)
@click.option("--dataset", type=str, required=True, help="Dataset key/name (passed into instance get_dataloader).")
@click.option("--num-classes", type=int, default=2, help="Includes background for Mask R-CNN style models.")
@click.option("--batch-size", type=int, default=4)
@click.option("--num-workers", type=int, default=8)
@click.option("--lr", type=float, default=1e-3)
@click.option("--epochs", type=int, default=30)
@click.option("--resume", type=click.Path(exists=True), default=None)
@click.option("--use-amp", is_flag=True)
@click.option("--gradient-clipping", type=float, default=0.1)
@click.option("--no-report", is_flag=True)
@click.option("--report-every", type=int, default=1)
def train_instance(**kwargs):
    """
    leaf-seg train instance ...
    """
    cfg = InstanceTrainConfig(**kwargs)
    instance_train_run(cfg)

# =============================================================================
# FINETUNE
# =============================================================================

# semantic finetune
@finetune.command("semantic")
@apply_opts(
    opt_device(),
    opt_progress(),
    opt_output("checkpoints/semantic/finetune"),
)
@click.option("--model", type=str, required=True)
@click.option("--encoder", type=str, required=True)
@click.option("--dataset", type=str, required=True)
@click.option("--num-classes", type=int, required=True)
@click.option("--ckpt", type=click.Path(exists=True), required=True, help="Pretrained checkpoint to load.")
@click.option("--batch-size", type=int, default=8)
@click.option("--num-workers", type=int, default=4)
@click.option("--lr", type=float, default=1e-4, help="Base LR (decoder_lr default).")
@click.option("--epochs", type=int, default=50)
@click.option("--use-amp", is_flag=True)
@click.option("--strict-load", is_flag=True, help="Require exact key/shape match when loading.")
@click.option("--freeze-encoder", is_flag=True, help="Freeze encoder for entire run.")
@click.option("--freeze-epochs", type=int, default=0, help="Freeze encoder for N epochs, then unfreeze.")
@click.option("--encoder-lr", type=float, default=None, help="Encoder LR (default: lr*0.1).")
@click.option("--decoder-lr", type=float, default=None, help="Decoder LR (default: lr).")
@click.option("--weight-decay", type=float, default=1e-4)
@click.option("--no-report", is_flag=True)
def finetune_semantic(**kwargs):
    """
    leaf-seg finetune semantic ...
    """
    cfg = SemanticFinetuneConfig(**kwargs)
    semantic_finetune_run(cfg)


# instance finetune
@finetune.command("instance")
@apply_opts(
    opt_device(),
    opt_progress(),
    opt_output("checkpoints/instance/finetune"),
)
@click.option("--dataset", type=str, required=True)
@click.option("--num-classes", type=int, required=True)
@click.option("--ckpt", type=click.Path(exists=True), required=True)
@click.option("--batch-size", type=int, default=8)
@click.option("--num-workers", type=int, default=4)
@click.option("--lr", type=float, default=1e-5)
@click.option("--epochs", type=int, default=20)
@click.option("--use-amp", is_flag=True)
@click.option("--strict-load", is_flag=True)
@click.option("--freeze-backbone", is_flag=True, help="Freeze encoder for entire run.")
@click.option("--freeze-epochs", type=int, default=0, help="Freeze encoder for N epochs, then unfreeze.")
@click.option("--backbone-lr", type=float, default=None, help="Encoder LR (default: lr*0.1).")
@click.option("--head-lr", type=float, default=None, help="Decoder LR (default: lr).")
@click.option("--weight-decay", type=float, default=1e-4)
@click.option("--score-threshold", type=float, default=0.00)
@click.option("--no-report", is_flag=True)
def finetune_instance(**kwargs):
    """
    leaf-seg finetune instance ...
    """
    cfg = InstanceFinetuneConfig(**kwargs)
    instance_finetune_run(cfg)


# =============================================================================
# EVAL
# =============================================================================

# eval semantic
@eval.command("semantic")
@apply_opts(
    opt_device(),
    opt_resize(default=(512, 512)),
    opt_output("results/semantic"),
)
@click.option("--model", required=True, type=str)
@click.option("--encoder", required=True, type=str)
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--dataset", required=True, type=str,
              help="Dataset registry key (from data/datasets.yaml).")
@click.option("--batch-size", type=int, default=8)
@click.option("--num-workers", type=int, default=4)
@click.option("--save-vis", is_flag=True, help="Save prediction visualisation montages.")
@click.option("--num-vis-samples", type=int, default=16, show_default=True,
              help="Number of samples to visualise (requires --save-vis).")
@click.option("--verbosity", "-v", count=True)
def eval_semantic(**kwargs):
    """
    leaf-seg eval semantic ...
    """
    cfg = SemanticEvalConfig(**kwargs)
    semantic_eval_run(cfg)


# eval instance
@eval.command("instance")
@apply_opts(
    opt_device(),
    opt_resize(default=None),
    opt_output("results/instance"),
)
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--dataset", required=True, type=str,
              help="Dataset registry key (from data/datasets.yaml).")
@click.option("--batch-size", type=int, default=4)
@click.option("--num-workers", type=int, default=4)
@click.option("--score-thresh", default=0.5, type=float)
@click.option("--iou-thresh", default=0.5, type=float)
@click.option("--save-vis", is_flag=True, help="Save prediction visualisation montages.")
@click.option("--num-vis-samples", type=int, default=16, show_default=True,
              help="Number of samples to visualise (requires --save-vis).")
@click.option("--verbosity", "-v", count=True)
def eval_instance(**kwargs):
    """
    leaf-seg eval instance ...
    """
    cfg = InstanceEvalConfig(**kwargs)
    instance_eval_run(cfg)


# =============================================================================
# INFER
# =============================================================================

# infer semantic
@infer.command("semantic")
@apply_opts(
    opt_device(),
    opt_resize(default=(512, 512)),
    opt_output("results/infer_semantic"),
)
@click.option("--model", required=True, type=str)
@click.option("--encoder", required=True, type=str)
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--num-classes", required=True, type=int)
@click.option("--image", type=click.Path(exists=True), default=None)
@click.option("--images", type=click.Path(exists=True), default=None, help="Folder of images.")
@click.option("--verbosity", "-v", count=True)
def infer_semantic(**kwargs):
    """
    leaf-seg infer semantic --image ... OR --images ...
    """
    # cfg = SemanticInferConfig(**kwargs)
    # semantic_infer_run(cfg)
    raise NotImplementedError("Wire semantic infer runner here.")


# infer instance
@infer.command("instance")
@apply_opts(
    opt_device(),
    opt_output("results/infer_instance"),
)
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--num-classes", required=True, type=int)
@click.option("--image", type=click.Path(exists=True), default=None)
@click.option("--images", type=click.Path(exists=True), default=None, help="Folder of images.")
@click.option("--score-thresh", default=0.5, type=float)
@click.option("--verbosity", "-v", count=True)
def infer_instance(**kwargs):
    """
    leaf-seg infer instance --image ... OR --images ...
    """
    # cfg = InstanceInferConfig(**kwargs)
    # instance_infer_run(cfg)
    raise NotImplementedError("Wire instance infer runner here.")


if __name__ == "__main__":
    cli()
