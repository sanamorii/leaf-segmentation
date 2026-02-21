import torch

from leaf_seg.models.maskrcnn_torch import get_model as get_maskrcnn


def setup_maskrcnn(num_classes: int, dataset: str, device: str) -> torch.nn.Module:
    model = get_maskrcnn(num_classes=num_classes)
    model.name = f"maskrcnn-{dataset}-instance"
    return model.to(torch.device(device))