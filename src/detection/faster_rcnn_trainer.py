import torch
import torch.utils.data
import os
import sys
import json
import math
from PIL import Image
from torchvision.transforms import v2 as T # Use v2 for modern transforms
from torch.optim.lr_scheduler import StepLR
from pathlib import Path # Import Path for checking files

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import local utility files
from src.detection.coco_utils import CocoDetection
from src.detection import engine
from src.detection import utils
# We must import calculate_mAP from evaluate.py
from src.detection.evaluate import calculate_mAP 

# Import model architecture
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# Import pre-trained weights
from torchvision.models.detection.weights import FasterRCNN_ResNet50_FPN_V2_Weights

# --- Configuration ---
DATASET_DIR = 'dataset' 
# The JSON files are in 'dataset/data_split/'
ANN_DIR = os.path.join(DATASET_DIR, 'data_split')
TRAIN_ANN_FILE = 'train_split.json'
VAL_ANN_FILE = 'val_split.json'

NUM_CLASSES = 2 # Anomaly (1) + Background (1)
NUM_EPOCHS = 25
BATCH_SIZE = 4
NUM_WORKERS = 4 # Use 0 if you get shared memory errors, 4 is faster
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
MODEL_SAVE_DIR = './models'
PRINT_FREQ = 50 # Log every 50 iterations

# --- Dataset Class (used for training) ---

class ThermalAnomalyDataset(CocoDetection):
    # This inherits from the CocoDetection in coco_utils.py
    def __init__(self, root, annFile, transforms=None):
        super().__init__(img_folder=root, ann_file=annFile, transforms=transforms)

    # We rely on the inherited __getitem__ from coco_utils.py
    pass


# --- Model Initialization ---

def get_model_instance_segmentation(num_classes):
    """Loads a pre-trained Faster R-CNN model (V2) and modifies the prediction head."""
    print("Initializing Faster R-CNN (ResNet-50-FPN-V2) using COCO Transfer Learning...")
    
    # Load the best available pre-trained weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# --- Main Training Function ---

def main():
    # Set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading datasets...")

    # --- CRITICAL PATH FIX HERE ---
    # The images are in 'dataset/train/images'
    # This path is now correct based on your 'ls' command.
    img_root = os.path.join(DATASET_DIR, 'train', 'images') 
    ann_root = os.path.join(DATASET_DIR, 'data_split')
    
    train_ann_path = os.path.join(ann_root, TRAIN_ANN_FILE)
    val_ann_path = os.path.join(ann_root, VAL_ANN_FILE)

    # Check if files exist before proceeding
    if not Path(img_root).exists():
        print(f"FATAL ERROR: Image directory not found at: {img_root}")
        print("Please check your 'dataset/train' folder and ensure an 'images' folder exists inside it.")
        return
    if not Path(train_ann_path).exists():
        print(f"FATAL ERROR: Training annotation file not found at: {train_ann_path}")
        return
        
    print(f"Loading training images from: {img_root}")
    
    dataset_train = ThermalAnomalyDataset(
        root=img_root, 
        annFile=train_ann_path,
        # We use the get_transform function
        transforms=get_transform(train=True) 
    )
    
    # DataLoaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=utils.collate_fn
    )

    print("DataLoaders created.")

    # --- Model Initialization ---
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(device)

    # --- Optimizer and LR Scheduler ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler which decreases the learning rate by 10x every 3 epochs
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # --- Training Loop ---
    print(f"Starting training for {NUM_EPOCHS} epochs on device: {device}")
    
    for epoch in range(NUM_EPOCHS): # Loop from 0 to 24
        # Training step
        engine.train_one_epoch(model, optimizer, data_loader_train, device, epoch, PRINT_FREQ)
        
        # Update the learning rate scheduler
        lr_scheduler.step()
        
        print(f"\nTraining complete for epoch {epoch + 1}.")
        
        # Save the model
        model_save_path = os.path.join(MODEL_SAVE_DIR, f"faster_rcnn_final_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved for Epoch {epoch + 1} to {model_save_path}\n")

        # --- Validation and mAP Calculation ---
        print(f"--- Running Full mAP Validation for Epoch {epoch + 1} ---")

        # Pass the correct paths to the evaluation function
        coco_stats = calculate_mAP(
            model_path=model_save_path, 
            dataset_dir=DATASET_DIR, # This is 'dataset'
            annotation_file=VAL_ANN_FILE, # This is 'val_split.json'
            device=device
        )
        
        if coco_stats:
            print(f"\n--- Epoch {epoch + 1} Validation Results ---")
            coco_stats.summarize()
            # The main mAP score is the first value in the stats array
            print(f"COCO AP@0.50:0.95: {coco_stats.stats[0]:.4f}")
        else:
            print(f"--- Epoch {epoch + 1} Validation Failed ---")
        
    print("\n\nTraining completed successfully!")

def get_transform(train):
    """Defines the data augmentations and conversion to tensor."""
    transforms = []
    # Convert PIL Image to PyTorch Tensor
    transforms.append(T.PILToTensor())
    # Convert to float and scale to [0, 1]
    transforms.append(T.ToDtype(torch.float, scale=True))
    
    if train:
        # Add standard augmentations for training
        transforms.append(T.RandomHorizontalFlip(0.5))
        # You can add more augmentations here later (e.g., color jitter for thermal)
    
    return T.Compose(transforms)

if __name__ == "__main__":
    # Check if the split files exist before starting
    ann_root_check = os.path.join(DATASET_DIR, 'data_split')
    train_ann_check = os.path.join(ann_root_check, TRAIN_ANN_FILE)
    val_ann_check = os.path.join(ann_root_check, VAL_ANN_FILE)

    if not Path(train_ann_check).exists() or not Path(val_ann_check).exists():
        print(f"FATAL ERROR: Annotation files not found in {ann_root_check}")
        print("Please run 'python3 src/tools/data_splitter.py' first!")
    else:
        main()