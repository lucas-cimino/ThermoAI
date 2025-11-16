import torch
import torch.utils.data
import os
import sys
import json # <--- ADD JSON IMPORT
import math
from PIL import Image
from torchvision.transforms import v2 as T
from torch.optim.lr_scheduler import StepLR
from pathlib import Path 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.detection.coco_utils import CocoDetection
from src.detection import engine
from src.detection import utils
from src.detection.evaluate import calculate_mAP 

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- Configuration ---
DATASET_DIR = 'dataset' 
ANN_DIR = os.path.join(DATASET_DIR, 'data_split')
TRAIN_ANN_FILE = 'train_split.json'
VAL_ANN_FILE = 'val_split.json'

NUM_CLASSES = 2 
NUM_EPOCHS = 25
BATCH_SIZE = 4
NUM_WORKERS = 4 
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
MODEL_SAVE_DIR = './models'
PRINT_FREQ = 50 

# --- Dataset Class (used for training) ---

class ThermalAnomalyDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(img_folder=root, ann_file=annFile, transforms=transforms)
    pass


# --- Model Initialization ---

def get_model_instance_segmentation(num_classes):
    print("Initializing Faster R-CNN (ResNet-50-FPN-V2) using COCO Transfer Learning...")
    # Use backward-compatible string for weights
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- Main Training Function ---

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading datasets...")
    img_root = os.path.join(DATASET_DIR, 'train', 'images') 
    ann_root = os.path.join(DATASET_DIR, 'data_split')
    train_ann_path = os.path.join(ann_root, TRAIN_ANN_FILE)
    val_ann_path = os.path.join(ann_root, VAL_ANN_FILE)

    if not Path(img_root).exists():
        print(f"FATAL ERROR: Image directory not found at: {img_root}")
        return
    if not Path(train_ann_path).exists():
        print(f"FATAL ERROR: Training annotation file not found at: {train_ann_path}")
        return
        
    print(f"Loading training images from: {img_root}")
    
    dataset_train = ThermalAnomalyDataset(
        root=img_root, 
        annFile=train_ann_path,
        transforms=get_transform(train=True) 
    )
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=utils.collate_fn
    )

    print("DataLoaders created.")

    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY
    )
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # --- NEW: Create a log collector ---
    metrics_log = []
    # ------------------------------------

    print(f"Starting training for {NUM_EPOCHS} epochs on device: {device}")
    
    for epoch in range(NUM_EPOCHS): # Loop from 0 to 24
        # --- MODIFIED: Capture the returned logger ---
        metric_logger = engine.train_one_epoch(model, optimizer, data_loader_train, device, epoch, PRINT_FREQ)
        
        lr_scheduler.step()
        
        model_save_path = os.path.join(MODEL_SAVE_DIR, f"faster_rcnn_final_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved for Epoch {epoch + 1} to {model_save_path}\n")

        print(f"--- Running Full mAP Validation for Epoch {epoch + 1} ---")
        coco_stats = calculate_mAP(
            model_path=model_save_path, 
            dataset_dir=DATASET_DIR, 
            annotation_file=VAL_ANN_FILE, 
            device=device
        )
        
        if coco_stats:
            print(f"\n--- Epoch {epoch + 1} Validation Results ---")
            coco_stats.summarize()
            map_score = coco_stats.stats[0] # Get mAP@.50:.95
            map_50_score = coco_stats.stats[1] # Get mAP@.50
            print(f"COCO AP@0.50:0.95: {map_score:.4f}")

            # --- NEW: Log the data ---
            epoch_data = {
                'epoch': epoch + 1,
                'loss': metric_logger.loss.global_avg, # Get final avg loss
                'mAP': map_score,
                'mAP_50': map_50_score
            }
            metrics_log.append(epoch_data)
            # ---------------------------
        else:
            print(f"--- Epoch {epoch + 1} Validation Failed ---")
            metrics_log.append({'epoch': epoch + 1, 'loss': None, 'mAP': None, 'mAP_50': None})
        
    print("\n\nTraining completed successfully!")

    # --- NEW: Save the log file ---
    log_path = os.path.join(MODEL_SAVE_DIR, 'training_metrics.json')
    print(f"Saving final metrics log to {log_path}")
    with open(log_path, 'w') as f:
        json.dump(metrics_log, f, indent=4)
    # ------------------------------

def get_transform(train):
    """Defines the data augmentations and conversion to tensor."""
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ToDtype(torch.float, scale=True))
    
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    return T.Compose(transforms)

if __name__ == "__main__":
    ann_root_check = os.path.join(DATASET_DIR, 'data_split')
    train_ann_check = os.path.join(ann_root_check, TRAIN_ANN_FILE)
    val_ann_check = os.path.join(ann_root_check, VAL_ANN_FILE)

    if not Path(train_ann_check).exists() or not Path(val_ann_check).exists():
        print(f"FATAL ERROR: Annotation files not found in {ann_root_check}")
        print("Please run 'python3 src/tools/data_splitter.py' first!")
    else:
        main()