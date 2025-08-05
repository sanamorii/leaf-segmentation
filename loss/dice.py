import torch
import torch.nn.functional as F


# i stole these off somewhere
def dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """Average of Dice coefficient for all batches, or for a single mask"""
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """Average of Dice coefficient for all classes"""
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
    """Dice loss (objective to minimize) between 0 and 1"""
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def mean_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    """
    Computes mean IoU across the batch.
    Assumes binary segmentation. Both preds and targets should be [B, 1, H, W].
    """
    preds = (preds > threshold).float()
    targets = (targets > 0.5).float()  # just in case

    intersection = (preds * targets).sum(dim=(-1, -2, -3))
    union = (preds + targets - preds * targets).sum(dim=(-1, -2, -3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    Computes pixel-wise accuracy for binary masks.
    """
    preds = (preds > threshold).float()
    targets = (targets > 0.5).float()
    correct = (preds == targets).float().sum()
    total = torch.numel(preds)
    return (correct / total).item()