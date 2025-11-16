import torch
import json
import os
import sys
from PIL import Image

# Import necessary dependencies from torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# Add the directory containing the project modules to the path
# NOTE: This ensures imports like 'src.detection.utils' work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import local utility files
from src.detection import coco_utils
from src.detection import coco_eval
from src.detection import engine
from src.detection import utils 

# --- Configuration ---
NUM_CLASSES = 2 # Anomaly (1) + Background (1)
DATA_DIR_DEFAULT = 'data/thermal_anomalies'
ANNOTATION_FILE_DEFAULT = 'validation.json' 
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --- Dataset Class ---
# We inherit directly from the custom CocoDetection class in coco_utils.py
# and rely on its __getitem__ for image and target loading, as it performs
# the correct COCO-to-PyTorch conversion for us.

class ThermalAnomalyDataset(coco_utils.CocoDetection):
    # NOTE: The __init__ of the base class (in coco_utils.py) expects 
    # (img_folder, ann_file, transforms). We must match this.
    def __init__(self, root, annFile, transforms=None):
        # The 'root' passed here is actually the path to the 'images' folder.
        super().__init__(img_folder=root, ann_file=annFile, transforms=transforms)
        self._transforms = transforms

    # We do NOT define __getitem__ or _convert_to_pytorch_target here, 
    # as the inherited methods from coco_utils.CocoDetection already
    # handle the image loading and annotation conversion into the
    # expected PyTorch target dictionary format.
    pass

# --- Model Initialization ---

def get_model_instance_segmentation(num_classes):
    """Loads a pre-trained Faster R-CNN model (V2) and modifies the prediction head."""
    model = fasterrcnn_resnet50_fpn_v2() 
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 3. Create the dataset and dataloader
    dataset_val = ThermalAnomalyDataset(
        root=os.path.join(dataset_dir, 'images'), 
        annFile=os.path.join(dataset_dir, 'annotations', annotation_file), 
        # Apply the transforms that simply convert the PIL image to a tensor
        transforms=lambda img, target: (F.to_tensor(img), target)
    )
    
    # Use the custom collate_fn from the correct 'utils' module
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn 
    )
    
    # 4. Run the evaluation using the engine's built-in function
    print("Starting validation...")
    
    # Set model to evaluation mode
    model.eval() 
    
    coco_stats = engine.evaluate(model, data_loader_val, device=device)
    
    return coco_stats

if __name__ == '__main__':
    # Example usage for standalone testing
    if not os.path.exists('models'):
        os.makedirs('models')
        
    dummy_model_path = './models/faster_rcnn_final_epoch_1.pth'

    if not os.path.exists(dummy_model_path):
        print(f"Warning: Dummy model file '{dummy_model_path}' not found. Please run the trainer first to generate it.")
        sys.exit(0) 
        
    print("--- Standalone mAP Test ---")
    
    stats = calculate_mAP(
        model_path=dummy_model_path,
        dataset_dir=DATA_DIR_DEFAULT, 
        annotation_file=ANNOTATION_FILE_DEFAULT, 
        device=DEVICE
    )
    
    print("\n--- COCO Evaluation Results ---")
    print(stats)