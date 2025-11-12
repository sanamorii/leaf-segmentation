"""
sam2_unet_full.py

Adapter variants: Adapter, ScaledAdapter, LowRankAdapter
RFBBlock
 - SAM2EncoderWrapper (ability to accept a real SAM2/Hiera backbone or a DummyBackbone)
 - SAM2UNet decoder with deep supervision heads
 - Weighted BCE + Weighted IoU loss for deep supervision
 - Example usage & tests

References:
 - Official SAM2 repo (installation / model): https://github.com/facebookresearch/sam2
 - SAM2-UNet reproduction / code: https://github.com/WZH0120/SAM2-UNet
 - HuggingFace SAM2 repos for pretrained weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import List, Optional

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


# ---------------------------
# Adapter variants
# ---------------------------
class Adapter(nn.Module):
    """Standard Adapter: 1x1 down -> GeLU -> 1x1 up -> GeLU with residual"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.down = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.act1 = nn.GELU()
        self.up = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.act2 = nn.GELU()
    def forward(self, x):
        y = self.down(x)
        y = self.act1(y)
        y = self.up(y)
        y = self.act2(y)
        return x + y


class ScaledResidualAdapter(nn.Module):
    """
    Scaled residual adapter:
      out = x + alpha * AdapterBlock(x)
    alpha is a learnable scalar initialised small (e.g., 1e-1) to stabilise fine-tuning.
    """
    def __init__(self, channels: int, reduction: int = 4, init_scale: float = 0.1):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.GELU()
        )
        # learnable scale
        self.alpha = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
    def forward(self, x):
        return x + self.alpha * self.block(x)


class LowRankAdapter(nn.Module):
    """
    Low-rank adapter that factorises the 1x1 projection into two smaller 1x1 convs:
      x -> W2( W1(x) ) with W1: C->r, W2: r->C
    This reduces parameter count while keeping capacity.
    """
    def __init__(self, channels: int, rank: int = 32):
        super().__init__()
        r = min(rank, max(1, channels // 2))
        self.proj_down = nn.Conv2d(channels, r, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.proj_up = nn.Conv2d(r, channels, kernel_size=1, bias=False)
    def forward(self, x):
        y = self.proj_down(x)
        y = self.act(y)
        y = self.proj_up(y)
        return x + y


# ---------------------------
# RFBBlock (Receptive Field Block)
# ---------------------------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RFBBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        inter = max(8, out_channels // 4)

        self.branch0 = nn.Sequential(BasicConv(in_channels, inter, kernel_size=1))
        self.branch1 = nn.Sequential(
            BasicConv(in_channels, inter, kernel_size=1),
            BasicConv(inter, inter, kernel_size=3, padding=1),
            BasicConv(inter, inter, kernel_size=3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_channels, inter, kernel_size=1),
            BasicConv(inter, inter, kernel_size=3, padding=1),
            BasicConv(inter, inter, kernel_size=3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_channels, inter, kernel_size=1),
            BasicConv(inter, inter, kernel_size=3, padding=1),
            BasicConv(inter, inter, kernel_size=3, padding=7, dilation=7),
        )

        self.conv_fuse = BasicConv(inter * 4, out_channels, kernel_size=3, padding=1)
        self.conv_res = BasicConv(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat([x0, x1, x2, x3], dim=1)
        out = self.conv_fuse(out)
        res = self.conv_res(x)
        return out + res


# ---------------------------
# SAM2EncoderWrapper: supports adapter_type, and accepts real or dummy backbones
# ---------------------------
class SAM2EncoderWrapper(nn.Module):
    def __init__(
        self,
        backbone,
        adapter_channels: List[int],
        rf_out_channels: int = 64,
        adapter_type: str = "standard",  # "standard", "scaled", "lowrank"
        freeze_backbone: bool = True,
        lowrank_rank: int = 32
    ):
        """
        backbone: model that returns 4 feature maps (list/tuple/dict)
        adapter_channels: list of channels for each of the 4 backbone outputs
        rf_out_channels: output channels after RFB
        adapter_type: which adapter variant to use
        """
        super().__init__()
        assert len(adapter_channels) == 4
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        adapter_cls = {
            "standard": lambda ch: Adapter(ch),
            "scaled": lambda ch: ScaledResidualAdapter(ch),
            "lowrank": lambda ch: LowRankAdapter(ch, rank=lowrank_rank)
        }[adapter_type]

        self.adapters = nn.ModuleList([adapter_cls(c) for c in adapter_channels])
        self.rfbs = nn.ModuleList([RFBBlock(c, rf_out_channels) for c in adapter_channels])
        self.rf_out_channels = rf_out_channels

    def forward(self, x):
        feats = self.backbone(x)
        # support dict, tuple, list
        if isinstance(feats, dict):
            feats = [feats[k] for k in sorted(feats.keys())]
        elif isinstance(feats, tuple):
            feats = list(feats)
        elif not isinstance(feats, list):
            raise ValueError("Backbone must return list/tuple/dict of 4 feature maps")

        if len(feats) != 4:
            raise ValueError(f"Backbone must return 4 features, got {len(feats)}")

        out_feats = []
        for i, f in enumerate(feats):
            f = self.adapters[i](f)
            f = self.rfbs[i](f)
            out_feats.append(f)
        return out_feats  # shallow -> deep order (E0 ... E3)


# ---------------------------
# Dummy backbone for local testing (produces 4 features)
# ---------------------------
class DummyBackbone(nn.Module):
    def __init__(self, in_channels=1, channels=[64, 128, 256, 512]):
        super().__init__()
        self.stages = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.stages.append(nn.Sequential(
                nn.Conv2d(prev, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            ))
            prev = ch
        self.pool = nn.MaxPool2d(2,2)
    def forward(self, x):
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
            x = self.pool(x)
        return feats  # E0..E3 (shallow->deep)


# ---------------------------
# SAM2UNet decoder with deep supervision
# ---------------------------
class SAM2UNet(nn.Module):
    def __init__(self, encoder_wrapper: SAM2EncoderWrapper, rf_out_channels: int = 64, out_channels: int = 1, deep_supervision: bool = True):
        """
        encoder_wrapper: SAM2EncoderWrapper
        rf_out_channels: channels after RFB (same for all stages)
        deep_supervision: if True, returns list of outputs [out_stage1, out_stage2, out_final]
        """
        super().__init__()
        self.encoder = encoder_wrapper
        self.rf_ch = rf_out_channels
        self.deep_supervision = deep_supervision

        # create 3 up steps
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.seg_heads = nn.ModuleList()  # for deep supervision: one head per up step

        for _ in range(3):
            self.up_convs.append(nn.ConvTranspose2d(self.rf_ch, self.rf_ch, kernel_size=2, stride=2))
            self.up_blocks.append(DoubleConv(self.rf_ch*2, self.rf_ch))
            self.seg_heads.append(nn.Conv2d(self.rf_ch, out_channels, kernel_size=1))

        # final head (after last up)
        self.final_head = nn.Conv2d(self.rf_ch, out_channels, kernel_size=1)

    def forward(self, x):
        enc_feats = self.encoder(x)  # list: [E0, E1, E2, E3] shallow->deep
        x = enc_feats[-1]  # E3
        outputs = []

        # Upsample steps: E3->E2, E2->E1, E1->E0 (3 steps)
        for idx in range(3):
            x = self.up_convs[idx](x)
            skip = enc_feats[-2 - idx]  # E2, E1, E0
            if x.shape[2:] != skip.shape[2:]:
                x = TF.resize(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.up_blocks[idx](x)
            # produce deep supervision output for this stage
            out_stage = self.seg_heads[idx](x)
            outputs.append(out_stage)

        final = self.final_head(x)
        if self.deep_supervision:
            # return outputs in order [stage1, stage2, final] or you can return all three intermediate+final
            # We'll return [outputs[0], outputs[1], final] for 3-level supervision (closest to paper)
            # But also include outputs[2] (before final head) if you want 4 supervision signals.
            return [outputs[0], outputs[1], final]
        else:
            return final


# ---------------------------
# Loss: Weighted BCE + Weighted IoU (works with logits)
# ---------------------------
def weighted_bce_iou_loss(pred_logits: torch.Tensor, target: torch.Tensor, weight_map: Optional[torch.Tensor] = None, eps: float = 1e-6):
    """
    pred_logits: (B,1,H,W) raw logits
    target: (B,1,H,W) binary {0,1} floats
    weight_map: (B,1,H,W) optional per-pixel weights (same shape)
    returns combined loss scalar
    """
    # BCE with logits
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, weight=weight_map, reduction='mean')
    # IoU (on probabilities)
    pred = torch.sigmoid(pred_logits)
    if weight_map is None:
        inter = (pred * target).sum(dim=[1,2,3])
        union = (pred + target - pred*target).sum(dim=[1,2,3])
    else:
        inter = (pred * target * weight_map).sum(dim=[1,2,3])
        union = ( (pred + target - pred*target) * weight_map ).sum(dim=[1,2,3])
    iou = (inter + eps) / (union + eps)
    iou_loss = 1.0 - iou.mean()
    # combine
    return bce + iou_loss


def deep_supervision_loss(preds: List[torch.Tensor], target: torch.Tensor, weights: Optional[List[float]] = None, weight_map: Optional[torch.Tensor] = None):
    """
    preds: list of logits at multiple scales: [s1_logits, s2_logits, final_logits]
    target: ground truth at original resolution
    weights: list of weights for each output (same len as preds). If None, equal weights.
    weight_map: optional per-pixel weight map at original resolution (resized to preds if needed)
    """
    if weights is None:
        weights = [1.0/len(preds)] * len(preds)
    assert len(weights) == len(preds)

    total_loss = 0.0
    for p, w in zip(preds, weights):
        # ensure p spatial dims match target; resize preds to target shape (logits)
        if p.shape[2:] != target.shape[2:]:
            p_resized = TF.resize(p, size=target.shape[2:], antialias=False)
        else:
            p_resized = p
        # if weight_map provided, resize it too
        wm = None
        if weight_map is not None:
            if weight_map.shape[2:] != p_resized.shape[2:]:
                wm = TF.resize(weight_map, size=p_resized.shape[2:], antialias=False)
            else:
                wm = weight_map
        loss_p = weighted_bce_iou_loss(p_resized, target, weight_map=wm)
        total_loss += w * loss_p
    return total_loss


# ---------------------------
# Helper to load SAM2 Hiera backbone if available
# ---------------------------
def load_sam2_hiera_backbone(model_size: str = "sam2-hiera-large", device: str = "cpu"):
    """
    Try to import and load a SAM2 Hiera backbone. This depends on the SAM2 repo version installed.
    If SAM2 is installed as a python package (e.g. pip install -e sam2), you may import Hiera directly.

    If import fails, this function raises ImportError and you can fallback to DummyBackbone.

    Example (after installing sam2): 
       from sam2.backbone.hiera import Hiera
       model = Hiera(...)
       checkpoint = torch.load("path_to_sam2_hiera_large.pth")
       model.load_state_dict(checkpoint["model"], strict=False)
    The exact names and checkpoint keys depend on the SAM2 release; consult:
     - https://github.com/facebookresearch/sam2
     - https://huggingface.co/facebook/sam2-hiera-large
    """
    try:
        # attempt to import a common Hiera path used in sam2 repos
        from sam2.backbone.hiera import Hiera  # type: ignore
        # create an instance; arguments depend on the sam2 release; these are example defaults
        # Please check the sam2 repo for correct config/kwargs.
        model = Hiera(img_size=1024, patch_size=4, in_chans=3)  # adjust as needed
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise ImportError(
            "Could not import SAM2 Hiera from local SAM2 install. "
            "Install SAM2 from https://github.com/facebookresearch/sam2 and try again. "
            f"Import error: {e}"
        )


# ---------------------------
# Example usage + tests
# ---------------------------
def example_swap_in_sam2_backbone_if_available():
    """
    This will try to create a real SAM2 Hiera backbone and wrap it. If not installed,
    it will use DummyBackbone so you can test everything locally.
    """
    use_real_sam2 = False
    try:
        # try to load real sam2 hiera
        hiera = load_sam2_hiera_backbone()
        # if successful, you must determine the names/sizes of returned features
        # For example Hiera-L might produce channels [144, 288, 576, 1152]
        adapter_channels = [144, 288, 576, 1152]
        encoder_wrapper = SAM2EncoderWrapper(backbone=hiera, adapter_channels=adapter_channels, rf_out_channels=64, adapter_type="scaled", freeze_backbone=True)
        use_real_sam2 = True
    except ImportError as ie:
        print("SAM2 not installed or could not be imported; falling back to dummy backbone for test.")
        print(ie)
        dummy_channels = [64, 128, 256, 512]
        dummy_backbone = DummyBackbone(in_channels=1, channels=dummy_channels)
        encoder_wrapper = SAM2EncoderWrapper(backbone=dummy_backbone, adapter_channels=dummy_channels, rf_out_channels=64, adapter_type="lowrank", freeze_backbone=False, lowrank_rank=32)

    model = SAM2UNet(encoder_wrapper, rf_out_channels=64, out_channels=1, deep_supervision=True)
    return model, use_real_sam2


def test_run():
    model, using_sam2 = example_swap_in_sam2_backbone_if_available()
    device = torch.device("cpu")
    model.to(device)
    # small test input
    x = torch.randn((2, 1, 161, 161), device=device)
    preds = model(x)
    # preds is a list of logits [stage1, stage2, final] (if deep_supervision True)
    assert isinstance(preds, list) and len(preds) == 3, "Expected 3 outputs for deep supervision"
    # test loss computation
    target = torch.randint(0,2,(2,1,161,161), dtype=torch.float32, device=device)
    loss = deep_supervision_loss(preds, target, weights=[0.2, 0.3, 0.5])
    print("Using real SAM2?" , using_sam2)
    print("Pred shapes:", [p.shape for p in preds])
    print("Loss:", loss.item())
    assert loss.item() > 0
    print("Test passed.")


if __name__ == "__main__":
    test_run()
