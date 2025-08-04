import torch
import torch.nn as nn
import segmentation_models_pytorch.losses as smp_losses

class CEDiceLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=255)  # Modify if you use a different ignore index
        self.dice = smp_losses.DiceLoss(mode='multiclass')  # For multiclass segmentation
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
