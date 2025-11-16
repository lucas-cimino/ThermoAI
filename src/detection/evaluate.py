import torch
import json
import os
import sys
from PIL import Image
from pathlib import Path

# Import necessary dependencies from torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T # Use v2 for modern transforms

# Add the directory containing the project modules to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import local utility files
from src.detection import coco_utils
from src.detection import coco_eval
from src.detection import engine
from src.detection import utils 

# --- Configuration ---
NUM_CLASSES = 2 
ANNOTATION_FILE_DEFAULT = 'val_split.json' 
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BATCH_SIZE = 4 # Match trainer batch size for consistency
NUM_WORKERS = 4

# --- Dataset Class ---
class ThermalAnomalyDataset(coco_utils.CocoDetection):
    # This inherits from the CocoDetection in coco_utils.py
    def __init__(self, root, annFile, transforms=None):
        super().__init__(img_folder=root, ann_file=annFile, transforms=transforms)

    # We rely on the inherited __getitem__ from coco_utils.py
    pass

# --- Model Initialization ---

def get_model_instance_segmentation(num_classes):
    """Loads a Faster R-CNN model (V2) architecture and modifies the prediction head."""
    # We load the architecture only, as we will load weights from our file
    model = fasterrcnn_resnet50_fpn_v2(weights=None) 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- Evaluation Function ---

def calculate_mAP(model_path, dataset_dir, annotation_file, device):
    """
    Loads a trained model, runs evaluation on the validation dataset, 
    and returns the COCO mAP statistics.
    """
    print(f"Loading model from: {model_path}")
    
    # 1. Initialize the model structure (V2)
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(device)
    
    # 2. Load the trained state dictionary
    try:
        # Use weights_only=True for security, as recommended by the warning
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except (TypeError, RuntimeError):
        # Fallback for older PyTorch versions or state_dict mismatch
        print("Note: weights_only=True failed. Loading with default.")
        model.load_state_dict(torch.load(model_path, map_location=device))

    
    # 3. Create the dataset and dataloader
    
    # --- CRITICAL PATH FIX HERE ---
    # The validation images are also in 'dataset/train/images'
    # (COCO datasets often use one 'train' folder for all images and split via JSON)
    img_root_path = os.path.join(dataset_dir, 'train', 'images') 
    ann_file_path = os.path.join(dataset_dir, 'data_split', annotation_file)
    
    print(f"Loading validation annotations from: {ann_file_path}")
    print(f"Loading validation images from: {img_root_path}")
    
    if not Path(img_root_path).exists():
        print(f"FATAL ERROR: Validation image directory not found at: {img_root_path}")
        return None
    if not Path(ann_file_path).exists():
        print(f"FATAL ERROR: Validation annotation file not found at: {ann_file_path}")
        return None

    dataset_val = ThermalAnomalyDataset(
        root=img_root_path, 
        annFile=ann_file_path, 
        # Apply the transforms that simply convert the PIL image to a tensor
        transforms=get_transform(train=False) # Use get_transform from trainer
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=utils.collate_fn 
    )
    
    # 4. Run the evaluation using the engine's built-in function
    print("Starting validation...")
    
    model.eval() 
    
    coco_stats = engine.evaluate(model, data_loader_val, device=device)
    
    return coco_stats

def get_transform(train):
    """Defines the data augmentations and conversion to tensor."""
    # Must be defined here as well for the standalone test
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ToDtype(torch.float, scale=True))
    
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    return T.Compose(transforms)


if __name__ == '__main__':
    # Example usage for standalone testing
    
    MODEL_SAVE_DIR = './models'
    DATASET_DIR = 'dataset'
    
    dummy_model_path = os.path.join(MODEL_SAVE_DIR, 'faster_rcnn_final_epoch_1.pth')

    if not Path(dummy_model_path).exists():
        print(f"Warning: Dummy model file '{dummy_model_param}' not found. Please run the trainer first to generate it.")
        sys.exit(0) 
        
    print("--- Standalone mAP Test ---")
    
    stats = calculate_mAP(
        model_path=dummy_model_path,
        dataset_dir=DATASET_DIR, 
        annotation_file=ANNOTATION_FILE_DEFAULT, 
        device=DEVICE
    )
    
    if stats:
        print("\n--- COCO Evaluation Results ---")
        stats.summarize()