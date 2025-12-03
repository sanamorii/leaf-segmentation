import time
import logging
import argparse
import datetime
import torch
import click

from pathlib import Path
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as smp_losses
from segmentation_models_pytorch.base.model import SegmentationModel
import matplotlib.pyplot as plt

from dataset.utils import rgb_to_class
from dataset.plantdreamer_semantic import get_dataloader
from loss.cedice import CEDiceLoss
from loss.earlystop import EarlyStopping
from models.utils import create_ckpt, save_ckpt, load_ckpt
from models.modelling import get_model
from metrics import StreamSegMetrics

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def setup_model(model_name, encoder, dataset, num_classes, device):
    # create a simple model and optimizer
    model = get_model(name=model_name, encoder=encoder, weights="imagenet", classes=num_classes)
    model.name = f"{model.__class__.__name__.lower()}-{encoder}-{dataset}"
    device = torch.device(device)
    return model.to(device)

def freeze_encoder(model):
    if hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False
        logger.info("Encoder frozen")
    else:
        logger.warning("Model has no encoder - freezing unsuccessful")


def validate_epoch(
    model: SegmentationModel,
    loader: DataLoader,
    metrics: StreamSegMetrics,
    loss_fn,
    device: str,
    epochs: tuple[int, int],
    verbose: bool = True,
):
    start = time.time()
    
    model.eval()
    metrics.reset()
    
    if verbose:
        val_bar = tqdm(loader, desc=f"Epoch {epochs[0]+1}/{epochs[1]} [Val]", leave=False)
    else:
        val_bar = loader

    running_vloss = 0
    batch_count = 0

    with torch.no_grad():
        for img, mask in val_bar:
            batch_count += 1
            img = img.to(device).float()
            mask = mask.to(device).long()

            output = model(img)
            # detaches tensor from comp graph, selects the highest score for each pixel
            # cpu() moves from gpu to cpu, numpy() converts from tensor to np array
            preds = output.detach().max(dim=1)[1].cpu().numpy()
            targets = mask.cpu().numpy()

            # update metrics and validation loss
            loss = loss_fn(output, mask)
            running_vloss += loss.item()
            metrics.update(targets, preds)
            if verbose: val_bar.set_postfix(loss=loss.item())

        score = metrics.get_results()
    
    elapsed_time = time.time() - start
    avg_val_loss = (running_vloss / batch_count) if batch_count > 0 else 0.0
    return elapsed_time, score, avg_val_loss


def train_epoch(
    model: SegmentationModel,
    loss_fn,
    optimiser: Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    loader: DataLoader,
    device: torch.device,
    epochs: tuple[int, int],
    use_amp: bool = False,           # mixed precision vs default precision (FP16)
    gradient_clipping: float = 1.0,  # prevent exploding gradients
    verbose : bool = True,
):
    start = time.time()

    model.train()
    running_loss = 0
    batch_count = 0

    if verbose:
        train_bar = tqdm(
            loader, desc=f"Epoch {epochs[0]+1}/{epochs[1]} [Train]", leave=False
        )
    else:
        train_bar = loader

    for imgs, masks in train_bar:
        batch_count += 1
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # print("Target min/max:", masks.min().item(), masks.max().item())

        optimiser.zero_grad(set_to_none=True)
        if use_amp:
            # autocast expects device_type to be either 'cuda' or 'cpu' (not 'cuda:0')
            device_type = 'cuda' if getattr(device, 'type', str(device)).startswith('cuda') else 'cpu'
            with torch.cuda.amp.autocast(device_type=device_type, enabled=True):
                outputs = model(imgs)
                loss = loss_fn(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            scaler.step(optimiser)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            
            loss.backward()
            optimiser.step()

        loss_item = loss.item()
        running_loss += loss_item
        if verbose: train_bar.set_postfix(loss=loss_item)
    
    elapsed_time = time.time() - start

    avg_loss = (running_loss / batch_count) if batch_count > 0 else 0.0
    return elapsed_time, avg_loss


def train_loop(
    model : SegmentationModel,
    loss_fn,
    optimiser,
    scheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs : int,
    device,
    num_classes : int,
    use_amp: bool = False,
    patience: int = 0,
    gradient_clipping: float = 0.1,
    # visualise: bool = False,
    monitor_metric: str = "Mean IoU",
    resume: str | None = None,
    start_epoch: int = 0,
):
    
    best_vloss = np.inf
    best_score = 0.0
    cur_itrs = 0

    # normalize device
    if not isinstance(device, torch.device):
        device = torch.device(device)

    # use the proper GradScaler constructor
    grad_scaler = torch.amp.GradScaler(device=device, enabled=use_amp)
    loss_stop_policy = EarlyStopping(patience=patience, delta=0.001)  # early stopping policy
    metrics = StreamSegMetrics(num_classes)

    # resume from checkpoint if provided
    if resume is not None:
        ckpt = load_ckpt(resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimiser.load_state_dict(ckpt["optimizer_state"])
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            logger.warning("Could not load scheduler state from checkpoint")
        cur_itrs = ckpt.get("cur_itrs", 0)
        best_vloss = ckpt.get("validation_loss", best_vloss)
        best_score = ckpt.get("mean_val_iou", best_score)
        start_epoch = int(ckpt.get("epoch", start_epoch))
        logger.info("Resumed training from %s (starting epoch=%d)", resume, start_epoch)

    for epoch in range(start_epoch, epochs):

        elapsed_ttime, avg_tloss = train_epoch(
            model=model,
            loss_fn=loss_fn,
            optimiser=optimiser,
            scaler=grad_scaler,
            loader=train_loader,
            device=device,
            epochs=(epoch, epochs),
            gradient_clipping=gradient_clipping,
            use_amp=use_amp
        )

        cur_itrs += len(train_loader) if hasattr(train_loader, '__len__') else 0

        elapsed_vtime, val_score, avg_vloss = validate_epoch(
            model=model,
            loader=val_loader,
            metrics=metrics,
            epochs=(epoch, epochs),
            loss_fn=loss_fn,
            device=device,
        )

        scheduler.step(avg_vloss)  # epoch param is deprecated

        logger.info(
            "Epoch %d/%d: Avg Train Loss: %.4f, Avg Val Loss: %.4f, Mean IoU: %.4f, Training time: %s, Validation time: %s",
            epoch+1,
            epochs,
            avg_tloss,
            avg_vloss,
            val_score.get(monitor_metric, 0.0),
            str(datetime.timedelta(seconds=int(elapsed_ttime))),
            str(datetime.timedelta(seconds=int(elapsed_vtime)))
        )

        # save model
        checkpoint = create_ckpt(
            cur_itrs=cur_itrs,
            model=model,
            optimiser=optimiser,
            scheduler=scheduler,
            tloss=avg_tloss,
            vloss=avg_vloss,
            vscore=val_score,
            epoch=epoch,
            num_classes=num_classes
        )

        # ensure checkpoints directory exists
        ckpt_dir = Path('checkpoints')
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_name = getattr(model, 'name', model.__class__.__name__)

        # update best_vloss and best_score based on monitor_metric
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
        metric_value = val_score.get(monitor_metric, None)
        if metric_value is not None and metric_value > best_score:
            best_score = metric_value
            save_ckpt(checkpoint, str(ckpt_dir / f"{model_name}-{epochs}_best.pth"))
        save_ckpt(checkpoint, str(ckpt_dir / f"{model_name}-{epochs}_current.pth"))

        torch.cuda.empty_cache()  # clear cache

        loss_stop_policy(best_vloss)
        if loss_stop_policy.early_stop:
            logger.info("No improvement - terminating.")
            break


@click.group()
def cli():
    """leaf-segmentation"""
    pass

@cli.command()
@click.option("--model", type=str, required=True, help="Name of the model to use.")
@click.option("--encoder", type=str, required=True, help="Name of the encoder to use")
@click.option("--dataset", type=str, required=True, help="Name of the dataset to use")
@click.option("--num_classes", type=int, required=True, help="Number of semantic classes")
@click.option("--batch_size", type=int, default=8)
@click.option("--num_workers", type=int, default=4)
@click.option("--lr", type=float, default=1e-3, help="Learning rate")
@click.option("--epochs", type=int, default=100, help="Number of epochs to train on.")
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu",
              help="Device to train the model on. Default is 'cuda' if available")
@click.option("--resume", type=click.Path(exists=True), default=None,
              help="Resume from a prior checkpoint")
@click.option("--use_amp", is_flag=True)
def train(model, encoder, dataset, num_classes, batch_size, num_workers, lr, epochs, device, resume, use_amp):
    # prepare dataloaders
    train_loader, val_loader = get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, num_classes=num_classes)
    model = setup_model(model, encoder, dataset, num_classes, device)

    # suggested by - https://arxiv.org/pdf/1206.5533
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.999], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=3)
    loss_fn = CEDiceLoss(ce_weight=0.5, dice_weight=0.5)

    train_loop(
        model=model,
        loss_fn=loss_fn,
        optimiser=optimiser,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        device=device,
        num_classes=num_classes,
        use_amp=use_amp,
        gradient_clipping=0.1,
        monitor_metric="Mean IoU",
        resume=resume,
    )



@cli.command()
def finetune(model, encoder, dataset, num_classes, batch_size, num_workers, lr, epochs, device, resume, use_amp):
    train_loader, val_loader = get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, num_classes=num_classes)

    pass

# CLI wrapper
if __name__ == "__main__":
    cli()

    # parser = argparse.ArgumentParser(description="Train segmentation model")
    # parser.add_argument("--dataset", help="dataset identifier for get_dataloader")
    # parser.add_argument("--model", type=str, help="model name")
    # parser.add_argument("--encoder", type=str, help="encoder name")
    # parser.add_argument("--epochs", type=int, default=50)
    # parser.add_argument("--batch-size", type=int, default=8)
    # parser.add_argument("--num-workers", type=int, default=4)
    # parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # parser.add_argument("--resume", default=None)
    # parser.add_argument("--use-amp", action='store_true')
    # parser.add_argument("--num-classes", type=int, default=3)
    # args = parser.parse_args()

    # # prepare dataloaders
    # train_loader, val_loader = get_dataloader(args.dataset, batch_size=args.batch_size, num_workers=args.num_workers, num_classes=args.num_classes)

    # # create a simple model and optimizer
    # model = get_model(name=args.model, encoder=args.encoder, weights="imagenet", classes=args.num_classes)
    # model.name = f"{model.__class__.__name__.lower()}-{args.encoder}-{args.dataset}"
    # device = torch.device(args.device)
    # model.to(device)

    # # suggested by - https://arxiv.org/pdf/1206.5533
    # optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=3)
    # loss_fn = CEDiceLoss(ce_weight=0.5, dice_weight=0.5)

    # train_fn(
    #     model=model,
    #     loss_fn=loss_fn,
    #     optimiser=optimiser,
    #     scheduler=scheduler,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     epochs=args.epochs,
    #     device=device,
    #     num_classes=args.num_classes,
    #     use_amp=args.use_amp,
    #     gradient_clipping=0.1,
    #     monitor_metric="Mean IoU",
    #     resume=args.resume,
    # )
