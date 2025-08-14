import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import functional as F

from models.maskrcnn_torch import get_model as maskrcnn
from dataset.instance import LeafDataset
from dataset.cvppp import CVPPPLeafDataset

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

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    ann_file = "data\\instance_train\\annotations.json"
    img_dir = "data\\instance_train\\gt"
    mask_dir = "data\\instance_train\\mask"
    dataset = LeafDataset(img_dir, mask_dir, ann_file, transforms=None)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    num_classes = len(dataset.coco.getCatIds()) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maskrcnn(num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        _inst_train_epoch(model, optimizer, train_loader, device, epoch)
        _inst_evaluate(model, val_loader, device)
        lr_scheduler.step()

    torch.save(model.state_dict(), "maskrcnn_coco.pth")

if __name__ == "__main__":
    main()