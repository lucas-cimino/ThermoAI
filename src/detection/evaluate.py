import torch
import torch.utils.data
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import evaluate as evaluate_model_func # Use the evaluation function from the engine script
from coco_utils import get_coco_api_from_dataset
from utils import collate_fn
from pathlib import Path
import json

# --- Configuration ---
NUM_CLASSES = 2 # Anomaly (1) + Background (1)
MODEL_PATH = 'models/faster_rcnn_final_epoch_25.pth'
DATA_DIR = 'data/thermal_anomalies'
# NOTE: We assume 'validation.json' is your COCO annotation file for evaluation
ANNOTATION_FILE = 'validation.json' 
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model_instance_segmentation(num_classes):
    """Loads a pre-trained Faster R-CNN model and modifies the prediction head."""
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained box predictor with one that knows about NUM_CLASSES
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_dataset_for_eval(data_dir, annotation_file):
    """Placeholder for loading the validation dataset. Needs to be implemented 
    based on your COCO dataset structure."""
    
    # IMPORTANT: You must create a class that inherits from torch.utils.data.Dataset
    # and loads your images and annotations (similar to what was done for training).
    # Since we don't have the full data loader class, we use a placeholder 
    # to demonstrate the evaluation flow.
    
    class ThermalAnomalyDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, annotation_file):
            # This must load your COCO JSON file and image paths
            # For demonstration, we just need the file path for the evaluator
            self.coco_json_path = Path(data_dir) / annotation_file

            # The evaluator needs the dataset object to have a 'coco' attribute 
            # that is an instance of the COCO API (pycocotools.coco.COCO)
            import pycocotools.coco as coco_api
            print(f"Loading COCO annotations from: {self.coco_json_path}")
            self.coco = coco_api.COCO(self.coco_json_path)
            self.img_ids = sorted(self.coco.imgs.keys())

        def __getitem__(self, idx):
            # The actual evaluation function in engine.py only needs to know the 
            # number of items. The COCO Evaluator takes care of the image processing.
            img_id = self.img_ids[idx]
            
            # Placeholder for image and target loading (not used in pure evaluation)
            # You would normally load the image and annotations here.
            # For this script, we return dummy tensors.
            image = torch.rand((3, 600, 800)) # Dummy image tensor
            target = {'image_id': torch.as_tensor(img_id), 'boxes': torch.empty((0, 4)), 'labels': torch.empty((0,), dtype=torch.int64)}

            return image, target

        def __len__(self):
            return len(self.img_ids)
            
    return ThermalAnomalyDataset(data_dir, annotation_file)


def calculate_mAP():
    """Calculates and prints the COCO mAP scores for the trained model."""
    
    print(f"Loading model from: {MODEL_PATH}")
    model = get_model_instance_segmentation(NUM_CLASSES)
    
    # Load the trained weights
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model file not found at {MODEL_PATH}. Cannot evaluate.")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    # Load the validation dataset
    try:
        dataset_test = load_dataset_for_eval(DATA_DIR, ANNOTATION_FILE)
    except Exception as e:
        print(f"\nFATAL ERROR: Could not load the validation dataset.")
        print("Please ensure 'data/thermal_anomalies/validation.json' exists and contains valid COCO annotations.")
        print(f"Original Error: {e}")
        return

    # Create the data loader
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )

    print(f"Starting evaluation on {len(dataset_test)} images...")
    
    # Run the evaluation function from engine.py
    # This function uses the pycocotools.cocoeval library
    coco_evaluator = evaluate_model_func(model, data_loader_test, device=DEVICE)

    # The COCO Evaluator prints the mAP scores (mAP@0.50:0.95 and mAP@0.50)
    print("\n--- COCO Mean Average Precision (mAP) Scores ---")
    coco_evaluator.coco_eval['bbox'].summarize()
    print("---------------------------------------------")

if __name__ == '__main__':
    calculate_mAP()