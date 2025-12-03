import torch
import torch.nn as nn
from torchvision.models import inception_v3

class InceptionV3Features(nn.Module):
    """
    Returns 2048-dim pool3 features from an ImageNet-pretrained Inception-v3.
    """

    def __init__(self):
        super().__init__()
        self.inception = inception_v3(weights="IMAGENET1K_V1", transform_input=False)
        self.inception.fc = nn.Identity()  # remove classification head
        self.inception.eval()

        for p in self.inception.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        # expecting x: [B, 3, 299, 299]
        logits = self.inception(x)
        # logits is the 2048-dim pooled feature
        return logits
