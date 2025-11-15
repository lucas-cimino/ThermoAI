import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import v2 as T
import os

# --- NEW IMPORTS (Now runnable since you are adding the files) ---
from coco_utils import CocoDetection 
from engine import train_one_epoch, evaluate
from utils import collate_fn 
# ----------------------------------------------------------------

# --- Configuration ---
NUM_CLASSES = 2  # 1 (anomaly) + 1 (background)
BATCH_SIZE = 4   
NUM_EPOCHS = 25  
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
PRINT_FREQ = 50 # Log every 50 iterations
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --- Data Paths (Updated based on 'dataset/train' folder) ---
IMAGE_DIR = './dataset/train/images' # Path to the image folder
TRAIN_JSON = './dataset/data_split/train_split.json'
VAL_JSON = './dataset/data_split/val_split.json'

# --- Output Path ---
MODEL_OUTPUT_DIR = './models'
# ---------------------

def get_transform(train):
    """Defines the data augmentations and conversion to tensor."""
    # Transforms will automatically run on the PIL Image loaded by CocoDetection
    transforms = [
        T.PILToTensor(),
        T.ToDtype(torch.float, scale=True),
    ]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    # We use T.Compose for a sequence of transformations
    return T.Compose(transforms)

def get_model(num_classes):
    """Initializes the Faster R-CNN model using COCO pre-trained weights."""
    print("Initializing Faster R-CNN (ResNet-50-FPN-V2) using COCO Transfer Learning...")
    
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    
    # Load the model with pre-trained weights for the backbone
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    # Modify the box predictor head for our specific number of classes (2: anomaly + background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)
    
    return model

def main():
    # 0. Setup directories
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")
    
    # 1. Load Datasets
    print("Loading datasets...")
    dataset_train = CocoDetection(IMAGE_DIR, TRAIN_JSON, transforms=get_transform(True))
    dataset_val = CocoDetection(IMAGE_DIR, VAL_JSON, transforms=get_transform(False))
    
    # 2. Create DataLoaders
    # collate_fn is ESSENTIAL for handling variable box counts in OD tasks
    data_loader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        collate_fn=collate_fn
    )
    data_loader_val = DataLoader(
        dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )
    print("DataLoaders created.")

    # 3. Initialize Model and Optimizer
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)
    
    # Optimization setup: Use SGD with momentum, common for object detection
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params, 
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    # 4. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs on device: {DEVICE}")
    for epoch in range(NUM_EPOCHS):
        # Train one epoch (loss logging handled inside engine.py)
        train_one_epoch(model, optimizer, data_loader_train, DEVICE, epoch, PRINT_FREQ)
        
        # Evaluate validation set (minimal check handled inside engine.py)
        evaluate(model, data_loader_val, DEVICE)

    # 5. Save Final Model
    save_path = os.path.join(MODEL_OUTPUT_DIR, f'faster_rcnn_final_epoch_{NUM_EPOCHS}.pth')
    print(f"\nTraining complete. Saving model state dict to {save_path}")
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    # Ensure data split runs successfully before trying to run the trainer!
    if not os.path.exists(TRAIN_JSON) or not os.path.exists(VAL_JSON):
        print("ERROR: Split JSON files not found. Please run src/tools/data_splitter.py first!")
    else:
        main()