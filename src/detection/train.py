import torch
import os
import json
import math
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T

# Add project root to path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.detection.coco_utils import CocoDetection, collate_fn
from src.detection.engine import train_one_epoch, evaluate

# Config
DATA_DIR = BASE_DIR / 'data' / 'train'
TRAIN_JSON = DATA_DIR / 'train_split.json'
VAL_JSON = DATA_DIR / 'val_split.json'
IMG_DIR = DATA_DIR / 'images'
MODEL_DIR = BASE_DIR / 'models'
LOG_FILE = MODEL_DIR / 'metrics.json'

BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 25
LR = 0.005
NUM_CLASSES = 2 

def get_model(num_classes):
    """Returns Faster R-CNN model with custom head."""
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_transform(train):
    """Returns transforms for training or validation."""
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ToDtype(torch.float, scale=True))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on {device}")
    
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Datasets
    dataset_train = CocoDetection(str(IMG_DIR), str(TRAIN_JSON), get_transform(train=True))
    dataset_val = CocoDetection(str(IMG_DIR), str(VAL_JSON), get_transform(train=False))

    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model = get_model(NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    history = []

    print("Starting training...")
    for epoch in range(EPOCHS):
        # Train
        logger = train_one_epoch(model, optimizer, loader_train, device, epoch, print_freq=50)
        lr_scheduler.step()
        
        # Validate
        coco_eval = evaluate(model, loader_val, device=device)
        
        # Stats
        stats = coco_eval.coco_eval['bbox'].stats
        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": logger.loss.global_avg,
            "val_map_50_95": stats[0],
            "val_map_50": stats[1],
            "val_recall_100": stats[8]
        }
        history.append(epoch_stats)
        
        # Save checkpoint
        torch.save(model.state_dict(), MODEL_DIR / f"model_epoch_{epoch+1}.pth")
        
        # Save Log
        with open(LOG_FILE, 'w') as f:
            json.dump(history, f, indent=4)

        print(f"Epoch {epoch+1} - Loss: {epoch_stats['train_loss']:.4f} | mAP: {epoch_stats['val_map_50_95']:.4f}")

if __name__ == '__main__':
    main()