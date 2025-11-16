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

# --- Configuration ---
NUM_CLASSES = 2 # Anomaly (1) + Background (1)
DATA_DIR = 'data/thermal_anomalies'
ANNOTATION_FILE = 'validation.json' 
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --- Dataset Class ---

class ThermalAnomalyDataset(coco_utils.CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        # We call the parent constructor from torchvision's CocoDetection
        super().__init__(root, annFile, transforms)
        self._transforms = transforms

    def __getitem__(self, idx):
        # Get the original image and target (annotation)
        img_id = self.ids[idx]
        target = self.coco.loadAnns(self.coco.getAnnIds(img_id))
        path = self.coco.loadImgs(img_id)[0]['file_name']
        
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        # Convert COCO format annotations to the expected PyTorch Detection format
        target = self._convert_to_pytorch_target(target, img.size)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def _convert_to_pytorch_target(self, coco_target, image_size):
        w, h = image_size
        
        # Process annotations for one image
        boxes = []
        labels = []
        iscrowd = []
        area = []

        for obj in coco_target:
            # COCO boxes are [x, y, w, h] - convert to [x_min, y_min, x_max, y_max]
            x_min = obj['bbox'][0]
            y_min = obj['bbox'][1]
            x_max = x_min + obj['bbox'][2]
            y_max = y_min + obj['bbox'][3]
            
            boxes.append([x_min, y_min, x_max, y_max])
            
            # The category ID is 1 for 'anomaly' (since we use 2 classes: 0=background, 1=anomaly)
            labels.append(1) 
            
            iscrowd.append(obj['iscrowd'])
            area.append(obj['area'])

        if not boxes:
            # Handle the case where an image has no annotations (should be rare/impossible for validation)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # Create the final target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([self.get_img_id(coco_target[0]['image_id'])])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.uint8)

        return target

# --- Model Initialization ---

def get_model_instance_segmentation(num_classes):
    """Loads a pre-trained Faster R-CNN model (V2) and modifies the prediction head."""
    # Use the V2 model to match the trainer!
    model = fasterrcnn_resnet50_fpn_v2() 
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one that knows our number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# --- Evaluation Function ---

# NOTE: Changing 'data_dir' to 'dataset_dir' to match the keyword argument used in faster_rcnn_trainer.py
def calculate_mAP(model_path, dataset_dir, ann_file, device):
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
        # Use dataset_dir instead of data_dir
        root=os.path.join(dataset_dir, 'images'), 
        annFile=os.path.join(dataset_dir, 'annotations', ann_file), 
        transforms=lambda img, target: (F.to_tensor(img), target)
    )
    
    # Use the custom collate_fn for object detection
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=coco_utils.collate_fn 
    )
    
    # 4. Run the evaluation using the engine's built-in function
    print("Starting validation...")
    coco_stats = engine.evaluate(model, data_loader_val, device=device)
    
    return coco_stats

if __name__ == '__main__':
    # Example usage for standalone testing
    
    # Create a dummy model directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Define a path for a model to test loading
    dummy_model_path = './models/faster_rcnn_final_epoch_1.pth'

    if not os.path.exists(dummy_model_path):
        print(f"Warning: Dummy model file '{dummy_model_path}' not found. Please run the trainer first to generate it.")
        sys.exit(0) 
        
    print("--- Standalone mAP Test ---")
    
    # Run the full evaluation process
    stats = calculate_mAP(
        model_path=dummy_model_path,
        dataset_dir=DATA_DIR, # Using DATA_DIR defined at the top
        ann_file=ANNOTATION_FILE,
        device=DEVICE
    )
    
    print("\n--- COCO Evaluation Results ---")
    print(stats)