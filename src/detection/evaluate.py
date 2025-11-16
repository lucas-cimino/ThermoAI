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

# --- Dataset Class ---
class ThermalAnomalyDataset(coco_utils.CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(img_folder=root, ann_file=annFile, transforms=transforms)
        self._transforms = transforms
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
    
    # FIX: Image root is assumed to be in the 'train' folder relative to the base dataset_dir
    img_root_path = os.path.join(dataset_dir, 'train') 
    ann_file_path = os.path.join(dataset_dir, 'data_split', annotation_file)
    
    print(f"Loading validation annotations from: {ann_file_path}")
    print(f"Loading validation images from: {img_root_path}") # This is the key path we are troubleshooting
    
    dataset_val = ThermalAnomalyDataset(
        root=img_root_path, 
        annFile=ann_file_path, 
        transforms=lambda img, target: (F.to_tensor(img), target)
    )
    
    # Use the custom collate_fn from the correct 'utils' module
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn 
    )
    
    # 4. Run the evaluation using the engine's built-in function
    print("Starting validation...")
    
    model.eval() 
    
    coco_stats = engine.evaluate(model, data_loader_val, device=device)
    
    return coco_stats

if __name__ == '__main__':
    # Example usage for standalone testing
    if not os.path.exists('models'):
        os.makedirs('models')
        
    dummy_model_path = './models/faster_rcnn_final_epoch_1.pth'
    DATA_DIR_DEFAULT = 'dataset'

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