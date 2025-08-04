from PIL import Image
import numpy as np
import albumentations as A

def preprocess_image(path, resize=(256,256)):
    image = Image.open(path).convert("RGB")
    trfm = A.Compose([
        A.Resize(resize),
        A.ToTensorV2(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return trfm(image).unsqueeze(0)



def infer(model, dataloader, device, threshold):
    return

def get_args():
    

def main():

    unetplusplus = smp.UnetPlusPlus(
        encoder_name=opts.encoder,
        encoder_weights=opts.weights,
        encoder_depth=5,
        in_channels=3,
        decoder_attention_type="scse",
        classes=len(COLOR_TO_CLASS),
    )
    return