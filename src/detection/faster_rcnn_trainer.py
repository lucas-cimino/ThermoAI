import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import v2 as T
import os
# NOTE: The imports below assume these utility files are placed in the same directory (src/detection/)
# from coco_utils import CocoDetection 
# from engine import train_one_epoch, evaluate
# import utils 

# --- Configuration ---
NUM_CLASSES = 2  # 1 (anomaly) + 1 (background)
BATCH_SIZE = 4   
NUM_EPOCHS = 25  
LEARNING_RATE = 0.005
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# --- Data Paths (Updated based on 'dataset/train' folder) ---
IMAGE_DIR = './dataset/train/images' # Path to the image folder
TRAIN_JSON = './dataset/data_split/train_split.json'
VAL_JSON = './dataset/data_split/val_split.json'
# --- Output Path ---
MODEL_OUTPUT_DIR = './models'
# ---------------------

def get_transform(train):
    """Defines the data augmentations (Minimal baseline for now)"""
    transforms = [
        T.PILToTensor(),
        T.ToDtype(torch.float, scale=True),
    ]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes):
    """Initializes the Faster R-CNN model using COCO pre-trained weights."""
    print("Initializing Faster R-CNN (ResNet-50-FPN-V2) using COCO Transfer Learning...")
    
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    
    # 1. Load the model with pre-trained weights for the backbone
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    # 2. Modify the box predictor head for your specific number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # We set num_classes to 2 (1 anomaly class + 1 background class)
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)
    
    return model

def main():
    # 0. Setup output directory
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")
    
    # Initialize the model using transfer learning
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)
    
    # Placeholder for the training loop logic which requires utility files
    print("--- DEPENDENCY CHECK ---")
    print("1. Run src/tools/data_splitter.py to create split JSONs.")
    print("2. Place 'coco_utils.py', 'engine.py', and 'utils.py' inside 'src/detection/' to enable imports.")
    print(f"Ready to start training for {NUM_EPOCHS} epochs with BATCH_SIZE={BATCH_SIZE}")

    # The actual training run would be here, using DataLoaders, Optimizer, and engine.py...

    # Placeholder save:
    save_path = os.path.join(MODEL_OUTPUT_DIR, f'faster_rcnn_final_epoch_{NUM_EPOCHS}.pth')
    # torch.save(model.state_dict(), save_path)
    print(f"Model will be saved to: {save_path}")

if __name__ == '__main__':
    main()