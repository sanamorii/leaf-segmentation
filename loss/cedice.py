import torch
import torch.nn as nn
import segmentation_models_pytorch.losses as smp_losses

class CEDiceLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=0) 
        self.dice = smp_losses.DiceLoss(mode='multiclass', classes=[1,2,3,4])
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
