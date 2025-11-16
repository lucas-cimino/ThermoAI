import torch
import torch.utils.data
import os
import sys
import json
import math
from PIL import Image
from torchvision.transforms import functional as F
from torch.optim.lr_scheduler import StepLR

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import local utility files
from src.detection.coco_utils import CocoDetection
from src.detection import engine
from src.detection import utils
from src.detection.evaluate import calculate_mAP

# Import model architecture
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- Configuration ---
# NOTE: Based on the file structure, the DATASET_DIR is 'dataset' and the 
# image sub-directory used for training/validation is 'dataset/train'.
DATASET_DIR = 'dataset' 
TRAIN_ANN_FILE = 'train_split.json'
VAL_ANN_FILE = 'val_split.json'

NUM_CLASSES = 2 # Anomaly (1) + Background (1)
NUM_EPOCHS = 25
BATCH_SIZE = 4
NUM_WORKERS = 4
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
MODEL_SAVE_DIR = './models'

# --- Dataset Class (used for training) ---

class ThermalAnomalyDataset(CocoDetection):
    # This transform function is crucial for training, as it converts the PIL image
    # to a tensor and keeps the target as is (the base class handles COCO-to-PyTorch conversion)
    def __init__(self, root, annFile, transforms=None):
        super().__init__(img_folder=root, ann_file=annFile, transforms=transforms)
        self._transforms = transforms

    # We rely on the inherited __getitem__ from CocoDetection in coco_utils.py
    # which performs the necessary image and target preparation.
    pass


# --- Model Initialization ---

def get_model_instance_segmentation(num_classes):
    """Loads a pre-trained Faster R-CNN model (V2) and modifies the prediction head."""
    print("Initializing Faster R-CNN (ResNet-50-FPN-V2) using COCO Transfer Learning...")
    # Load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one for our number of classes
    # (num_classes includes the background class, hence 2 for our case: background + anomaly)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# --- Main Training Function ---

def main():
    # Set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading datasets...")

    # Define paths
    img_root = os.path.join(DATASET_DIR, 'train') 
    ann_dir = os.path.join(DATASET_DIR, 'data_split')
    
    dataset_train = ThermalAnomalyDataset(
        root=img_root, 
        annFile=os.path.join(ann_dir, TRAIN_ANN_FILE),
        # Training transformation: PIL to Tensor
        transforms=lambda img, target: (F.to_tensor(img), target) 
    )
    
    # The validation dataset uses the same class, but is primarily used by the evaluate.py
    # The trainer needs to know the files exist, but evaluate.py handles its own dataloader.
    
    # DataLoaders
    # The collate_fn is crucial for combining targets in batches
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=utils.collate_fn
    )

    print("DataLoaders created.")

    # --- Model Initialization ---
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(device)

    # --- Optimizer and LR Scheduler ---
    # Separate parameters for different parts of the model (e.g., backbone vs. heads)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY
    )

    # Use StepLR for learning rate reduction
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    # Create model directory if it doesn't exist
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # --- Training Loop ---
    print(f"Starting training for {NUM_EPOCHS} epochs on device: {device}")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training step
        engine.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=50)
        
        # Update the learning rate scheduler
        lr_scheduler.step()
        
        print("Training complete for epoch.")
        
        # Save the model
        model_save_path = os.path.join(MODEL_SAVE_DIR, f"faster_rcnn_final_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved for Epoch {epoch} to {model_save_path}\n")

        # --- Validation and mAP Calculation ---
        print(f"--- Running Full mAP Validation for Epoch {epoch} ---")

        # IMPORTANT: Pass the base directory and the specific validation annotation filename
        # This will be used by evaluate.py to construct the full path: dataset/data_split/val_split.json
        coco_stats = calculate_mAP(
            model_path=model_save_path, 
            dataset_dir=DATASET_DIR, 
            annotation_file=VAL_ANN_FILE, # Pass 'val_split.json'
            device=device
        )
        
        print(f"\n--- Epoch {epoch} Validation Results ---")
        # Print the relevant COCO metrics (e.g., Average Precision (AP) @ IoU=0.50:0.95)
        # The engine.py returns a dictionary of metrics, print them nicely.
        coco_stats.summarize()
        # Optionally, print the main AP metric
        print(f"COCO AP@0.50:0.95: {coco_stats.stats[0]:.4f}")
        
    print("\n\nTraining completed successfully!")

if __name__ == "__main__":
    main()