import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import functional as F

from models.maskrcnn_torch import get_model as maskrcnn
from dataset.instance import LeafDataset
from dataset.cvppp import CVPPPLeafDataset

from pycocotools.cocoeval import COCOeval
import numpy as np

def _inst_train_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    running_loss = 0
    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        running_loss += losses.item()
        if i % print_freq == 0:
            print(f"Epoch [{epoch}] Iter [{i}/{len(data_loader)}] Loss: {losses.item():.4f}")
    return running_loss / len(data_loader)


def _inst_evaluate(model, data_loader, device):
    model.eval()
    loss_sum = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_sum += losses.item()
    print(f"Validation Loss: {loss_sum / len(data_loader):.4f}")
    return loss_sum / len(data_loader)

def _convert_to_coco_format(predictions):
    coco_results = []
    for pred in predictions:
        boxes = pred["boxes"].tolist()
        scores = pred["scores"].tolist()
        labels = pred["labels"].tolist()
        image_id = pred["image_id"]

        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            coco_results.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x_min, y_min, width, height],
                "score": float(score)
            })
    return coco_results


def _inst_evaluate(model, data_loader, device, coco_gt=None):
    model.eval()
    val_loss_sum = 0.0
    predictions = []
    targets_all = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model.train()  # force loss computation
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss_sum += losses.item()

            model.eval()
            outputs = model(images)  # list of {boxes, labels, scores, masks}
            for t, o in zip(targets, outputs):
                targets_all.append({
                    "image_id": int(t["image_id"].item()),
                    "boxes": t["boxes"].cpu(),
                    "labels": t["labels"].cpu(),
                    "masks": t["masks"].cpu()
                })
                predictions.append({
                    "image_id": int(t["image_id"].item()),
                    "boxes": o["boxes"].cpu(),
                    "labels": o["labels"].cpu(),
                    "scores": o["scores"].cpu(),
                    "masks": o["masks"].cpu()
                })

    avg_val_loss = val_loss_sum / len(data_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")


    if coco_gt is not None:
        coco_dt = coco_gt.loadRes(_convert_to_coco_format(predictions))
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")  # "bbox" for box mAP
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    # else:
    #     print("Skipping mAP calculation (no COCO ground truth provided)")

    return avg_val_loss


def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    ann_file = "data/instance_train/annotations.json"
    img_dir = "data/instance_train/gt"
    mask_dir = "data/instance_train/mask"

    dataset = LeafDataset(img_dir, mask_dir, ann_file, transforms=None)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)

    num_classes = len(set(dataset.cat_id_to_class_idx.values()))  # includes background

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maskrcnn(num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.00046399800282455436, weight_decay=3.0835541457599805e-07,
            betas=(0.847624784863984, 0.9417463044535068),
            eps=1.9789670128284147e-07
        )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.1118140163664775
        )

    num_epochs = 200
    for epoch in range(num_epochs):
        _inst_train_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = _inst_evaluate(model, val_loader, device)
        lr_scheduler.step(val_loss)

    torch.save(model.state_dict(), "maskrcnn_coco.pth")

if __name__ == "__main__":
    main()