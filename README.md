# Leaf Segmentation
Plant phenotyping tasks, such as image segmentation, have been vital towards the study of phenotypic traits. Much of the food and agriculture industry heavily rely on this research as plant diseases often can cause damages up to the billions, however a common challenge to solving these tasks is the amount of quality training data available. An approach to this would be to generate synthetic data based on known plant traits to help supplement training; research such as PlantDreamer has seen substantial improvements in the ability to generate synthetic 3D objects and 2D images using AI, although not much further research has been made to put this in context of said tasks. This project aims to discover new findings and limitations associated with using synthetic data in the context of plant phenotyping tasks. 


# Usage
This project was trained on x4 RTX A4000 GPUS provided by the University of Nottingham School of Computer Science and Google's Research Cloud TPUs.

Install prerequistes first - it's recommended that you create an environment for this
```
python -m venv segmleaf
source ./segmleaf/bin/activate
pip install -r requirements
```

```
python train.py
```

```
python predict.py
```

```
python validate.py
```

# Future Work
- Custom ViT implementation (explore finetuning SAM2?)
- Extend work onto leaf disease segmentation
- PlantDreamer doesn't produce realistic textures which I think  

# Reference
- [DeeplabV3Plus]()
- [DeeplabV3]()
- [Fast RCNN]()
- [Mask RCNN]()
- [SAM2]()
- 
