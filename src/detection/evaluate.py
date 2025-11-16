import torch
import torch.utils.data
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# Import the evaluation function (assuming you successfully merged the fix for coco_utils)
from engine import evaluate as evaluate_model_func 
from utils import collate_fn
from pathlib import Path
import json

# --- Configuration ---
NUM_CLASSES = 2 # Anomaly (1) + Background (1)
MODEL_PATH = 'models/faster_rcnn_final_epoch_25.pth'
DATA_DIR = 'data/thermal_anomalies'
ANNOTATION_FILE = 'validation.json' 
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model_instance_segmentation(num_classes):
    """Loads a pre-trained Faster R-CNN model and modifies the prediction head."""
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Placeholder for your specific dataset class (Must be consistent with train.py)
class ThermalAnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, annotation_file):
        import pycocotools.coco as coco_api
        self.coco_json_path = Path(data_dir) / annotation_file
        
        if not self.coco_json_path.exists():
             raise FileNotFoundError(f"Annotation file not found at {self.coco_json_path}. Cannot initialize COCO evaluator.")
             
        self.coco = coco_api.COCO(self.coco_json_path)
        self.img_ids = sorted(self.coco.imgs.keys())

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        # Dummy tensors required by the DataLoader, though the evaluator primarily uses self.coco
        image = torch.rand((3, 600, 800)) 
        target = {'image_id': torch.as_tensor(img_id), 'boxes': torch.empty((0, 4)), 'labels': torch.empty((0,), dtype=torch.int64)}
        return image, target

    def __len__(self):
        return len(self.img_ids)

def calculate_mAP(model_path, dataset_dir, annotation_file, device):
    """Calculates and returns COCO metrics and the custom composite score."""
    
    model = get_model_instance_segmentation(NUM_CLASSES)
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}. Cannot evaluate.")
        return None
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    try:
        dataset_test = ThermalAnomalyDataset(dataset_dir, annotation_file)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        return None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    print(f"\nStarting evaluation on {len(dataset_test)} images...")
    
    # Run the evaluation function, which returns the COCO Evaluator object
    coco_evaluator = evaluate_model_func(model, data_loader_test, device=device)
    
    # --- Extracting COCO Metrics ---
    coco_metrics = coco_evaluator.coco_eval['bbox'].stats
    
    # COCO Metrics Index:
    # 0: AP @ IoU=0.50:0.95 (mAP)
    # 1: AP @ IoU=0.50 
    # 8: AR @ IoU=0.50:0.95 (Max Detections = 100) -> Used as Recall proxy
    # We will use the AP at 0.50 as a Precision proxy for simplicity, as Precision/Recall are usually calculated 
    # across confidence thresholds, but AP@0.50 is the standard single-value metric.
    
    mAP50_95 = coco_metrics[0]
    mAP50 = coco_metrics[1] # Use this as the Precision proxy
    Recall_Proxy = coco_metrics[8] # AR Max Detections=100 as Recall proxy

    # --- Custom Score Calculation ---
    # Score = (mAP50-95 * 0.8) + (Precision * 0.1) + (Recall * 0.1)
    
    # Note: We are using mAP@0.50 as the Precision Proxy
    custom_score = (mAP50_95 * 0.8) + (mAP50 * 0.1) + (Recall_Proxy * 0.1)
    
    results = {
        'mAP50_95': mAP50_95,
        'mAP50': mAP50,
        'Recall_Proxy': Recall_Proxy,
        'Custom_Score': custom_score
    }
    
    print("\n--- Final Evaluation Summary ---")
    print(f"mAP@0.50:0.95 (80% Weight): {mAP50_95:.4f}")
    print(f"mAP@0.50 (Precision Proxy 10% Weight): {mAP50:.4f}")
    print(f"AR@100 (Recall Proxy 10% Weight): {Recall_Proxy:.4f}")
    print(f"🔥 Custom Composite Score: {custom_score:.4f} 🔥")
    print("--------------------------------")
    
    return results

if __name__ == '__main__':
    calculate_mAP(MODEL_PATH, DATA_DIR, ANNOTATION_FILE, DEVICE)