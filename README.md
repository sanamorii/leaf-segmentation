# Leaf Segmentation
Plant phenotyping tasks, such as image segmentation, have been vital towards the study of phenotypic traits. Much of the food and agriculture industry heavily rely on this research as plant diseases often can cause damages up to the billions, however a common challenge to solving these tasks is the amount of quality training data available. An approach to this would be to generate synthetic data based on known plant traits to help supplement training; research such as PlantDreamer has seen substantial improvements in the ability to generate synthetic 3D objects and 2D images using AI, although not much further research has been made to put this in context of said tasks. This project aims to discover new findings and limitations associated with using synthetic data in the context of plant phenotyping tasks. 

# Available Models
- DeeplabV3Plus
- DeeplabV3
- UNET
- UNET++
- UNET with Dropout (Custom)
- FPN
- Mask2Former
- SAM2 Encoder w/ UNET (custom/wip)
- SAM2 Encoder w/ DeepLab (custom/wip)
- SAM2 (instance segmentation/wip)
- Mask RCNN (instance segmenetation/wip)

- resnet
- mobilenet
- efficientnet

I've tried to make the codebase as modular as possible so implementing custom/other models should be possible.

# Usage
This project was trained on RTX A4000 GPUS provided by the University of Nottingham School of Computer Science and Google's Research Cloud TPUs.

Install prerequistes first - it's recommended that you create an environment for this
```
python -m venv segmleaf
source ./segmleaf/bin/activate
pip install -r requirements
```

```
python train.py
```

## Inference
```
python predict.py
```


## Evaluation
```
usage: validate.py [-h] --images IMAGES --masks MASKS --checkpoint CHECKPOINT [--output OUTPUT] [--resize RESIZE RESIZE] [--device DEVICE]

Evaluate segmentation model on image folder

options:
  -h, --help            show this help message and exit
  --images IMAGES       Path to folder containing input images
  --masks MASKS         Path to folder containing ground truth masks
  --checkpoint CHECKPOINT
                        Path to trained model checkpoint
  --output OUTPUT       Folder to save prediction results
  --resize RESIZE RESIZE
                        Resize shape: height width
  --device DEVICE       Device to run inference on
```

Mean IoU, Dice Coefficient, and Pixel Accuracy will be displayed after evaluation.

# Future Work
- Custom ViT implementation (explore finetuning SAM2?)
- Extend work onto leaf disease segmentation
- PlantDreamer doesn't produce realistic textures which I think  

# Reference
- [DeeplabV3Plus]()
- [DeeplabV3]()
- [UNET]()
- [UNET++]()
- [Fast RCNN]()
- [Mask RCNN]()
- [SAM2]()
