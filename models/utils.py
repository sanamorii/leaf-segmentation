import os
import torch
import logging

from segmentation_models_pytorch.base.model import SegmentationModel

def create_ckpt(
    cur_itrs: int, 
    model : SegmentationModel, 
    num_classes : int,
    optimiser, 
    scheduler, 
    tloss, 
    vloss, 
    vscore, 
    epoch: int = None
):
    ckpt = {
        "cur_itrs": cur_itrs,
        "model_name": model.name,
        "model_state": model.state_dict(),
        "optimizer_state": optimiser.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "validation_loss": vloss,
        "training_loss": tloss,
        "overall_val_acc": vscore["Overall Acc"],
        "mean_val_acc": vscore["Mean Acc"],
        "freqw_val_acc": vscore["FreqW Acc"],
        "mean_val_iou": vscore["Mean IoU"],
        "class_val_iou": vscore["Class IoU"],
        "num_classes": num_classes
    }
    if epoch is not None:
        ckpt["epoch"] = int(epoch)
    return ckpt


def save_ckpt(checkpoint, path):
    """ save current model with safe directory creation and logging
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(checkpoint, path)
    logging.getLogger(__name__).info("Model saved as %s", path)


def load_ckpt(path, map_location=None):
    """Load a checkpoint file and return the dict. Uses torch.load.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=map_location)
    logging.getLogger(__name__).info("Loaded checkpoint %s", path)
    return ckpt
