## DeepLabV3Plus_Resnet151_Imagenet
```
0: (0, 0, 0),       # Background (black)
1: (0, 255, 0),     # Leaf (green)
2: (255, 165, 0),   # Pot (orange)
3: (139, 69, 19),   # Soil (brown)
4: (157, 0, 255),   # Stem (purple)
```
![[Untitled 1.png]]
bean5/gt/1.png
![[Untitled-1.png]]
![[bean5-1-img.png]]

bean5/gt/2.png
![[bean5-2-img.png]]
![[Untitled-1 2.png]]

bean5/gt/3.png
![[Untitled 4.png]]
![[Untitled-1 3.png]]

bean5/gt/4.png
![[Untitled-1 5.png]]
![[Untitled 6.png]]

bean5/gt/5.png
![[Untitled-1 6.png]]
![[Untitled 7.png]]

bean5/gt/6.png
![[Untitled-1 7.png]]
![[Untitled 8.png]]

bean5/gt/7.png
![[Untitled-1 8.png]]
![[Untitled 9.png]]

bean5/gt/8.png
![[Untitled-1 9.png]]
![[Untitled 10.png]]

bean5/gt/9.png
![[Untitled-1 10.png]]
![[Untitled 11.png]]

bean5/gt/10.png
![[Untitled-1 11.png]]
![[Untitled 12.png]]

Evaluation Metrics:
Mean IoU      : 0.8969
Mean F1 Score : 0.9432
Mean Accuracy : 0.9436

```json
{'iou': 0.8969278931617737, 'f1': 0.9432175159454346, 'accuracy': 0.9436219930648804}
```

```json
{0: 'Background', 1: 'Leaf', 2: 'Pot', 3: 'Soil', 4: 'Stem'}
```
![[Untitled.png]]

## Real Data Inference
![[Untitled 14.png]]
![[Untitled-1 12.png]]
![[Untitled 15.png]]
![[Untitled-1 13.png]]
![[Untitled 16.png]]
![[Untitled-1 14.png]]
![[Untitled 17.png]]

### Some issues
- Imbalance with the background class; remove this in next model
- no metrics for the real bean data; need to annotate them by hand
	- really noisy/incorrect data for the real bean data predictions
	- maybe it's overfitting? need to check on real data first
- replace loss function with dice coefficient instead of cross entropy
- i need more compute :(