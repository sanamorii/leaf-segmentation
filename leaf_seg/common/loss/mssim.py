import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM, MS_SSIM

class MultiClassSSIMLoss(nn.Module):
    """
    Multi-class SSIM or MS-SSIM loss.
    pred: BxCxHxW (logits or probabilities)
    target: BxHxW (class ids) or BxCxHxW (one-hot)
    """
    def __init__(self, use_ms_ssim=False, data_range=1.0, win_size=11, win_sigma=1.5):
        super().__init__()
        LossClass = MS_SSIM if use_ms_ssim else SSIM
        # channel=1 because we apply per-class SSIM
        self.ssim = LossClass(
            data_range=data_range,
            win_size=win_size,
            win_sigma=win_sigma,
            channel=1,
            size_average=True
        )

    def forward(self, pred, target):
        # if pred is logits -> softmax
        if pred.dtype == torch.float32:
            pred = F.softmax(pred, dim=1)

        # ensure target is one-hot
        if target.dim() == 3:
            # B,H,W -> B,C,H,W
            target = F.one_hot(target, num_classes=pred.size(1))
            target = target.permute(0, 3, 1, 2).float()

        # compute per-class SSIM and average
        losses = []
        C = pred.size(1)

        for c in range(C):
            p_c = pred[:, c:c+1]
            t_c = target[:, c:c+1]
            loss_c = 1 - self.ssim(p_c, t_c)
            losses.append(loss_c)

        return torch.mean(torch.stack(losses))
