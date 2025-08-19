import segmentation_models_pytorch as smp
from .unetdropout import UNETDropout

MODEL_CHOICES = [
    "segformer",
    "unet",
    "unetplusplus",
    "unetdropout",
    "fpn",
    "deeplabv3plus",
    "deeplabv3",
]
ENCODER_CHOICES = [
    "mit_b0",
    "mit_b1",
    "mit_b2",
    "mit_b3",
    "mit_b4",
    "mit_b5",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",
    "efficientnet-b5",
    "efficientnet-b6",
    "efficientnet-b7",
]

def get_model(name: str, encoder: str, weights: str, classes: int):
    if name == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=weights,
            encoder_depth=5,
            in_channels=3,
            # decoder_channels=[128, 64, 32, 16, 8],  # [256, 128, 64, 32, 16]
            decoder_attention_type="scse",
            classes=classes,
        )
    elif name == "unet":
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            encoder_depth=5,
            in_channels=3,
            # decoder_channels=[128, 64, 32, 16, 8],  # [256, 128, 64, 32, 16]
            classes=classes,
        )
    elif name == "unetdropout":
        return UNETDropout(
            encoder_name=encoder,
            encoder_weights=weights,
            encoder_depth=5,
            in_channels=3,
            num_classes=classes,
        )
    elif name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=classes,
        )
    elif name == "deeplabv3":
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=classes,
        )
    elif name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=classes,
        )
    elif name == "segformer":
        return smp.Segformer(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=classes,
        )

