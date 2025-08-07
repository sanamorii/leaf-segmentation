import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))

        layers += [
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UNETDropout(nn.Module):
    def __init__(self, encoder: str = "resnet34", weights: str = None, in_channels=3, encoder_depth=5, num_classes=1, dropout=0.3):
        super(UNETDropout, self).__init__()
        self.dropout = dropout

        
        self.encoder = smp.encoders.get_encoder(
            encoder,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=weights,
        )
        encoder_channels = self.encoder.out_channels  # e.g. [64, 64, 128, 256, 512]

        self.bottleneck = DoubleConv(encoder_channels[-1], encoder_channels[-1] * 2, dropout=dropout)

        self.ups = nn.ModuleList()
        for i in range(len(encoder_channels) - 1, 0, -1):
            upconv_in_channels = encoder_channels[i] * 2 if i == len(encoder_channels) - 1 else encoder_channels[i]
            upconv_out_channels = encoder_channels[i - 1]
            self.ups.append(
                nn.ConvTranspose2d(upconv_in_channels, upconv_out_channels, kernel_size=2, stride=2)
            )
            self.ups.append(
                DoubleConv(upconv_out_channels + encoder_channels[i - 1], upconv_out_channels, dropout=dropout)
            )

        self.final_conv = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        enc_feats = self.encoder(x)
        bottleneck = self.bottleneck(enc_feats[-1])
        x = bottleneck
        up_idx = 0

        for i in range(len(enc_feats) - 1, 0, -1):
            upconv = self.ups[up_idx]
            doubleconv = self.ups[up_idx + 1]
            up_idx += 2

            x = upconv(x)
            x = torch.cat([x, enc_feats[i - 1]], dim=1)
            x = doubleconv(x)

        return self.final_conv(x)