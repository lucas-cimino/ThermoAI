import matplotlib.pyplot as plt
import json
import torch
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Config
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / 'models'
LOG_FILE = MODEL_DIR / 'metrics.json'
OUTPUT_DIR = BASE_DIR / 'plots'
TEST_IMG_DIR = BASE_DIR / 'data' / 'test' / 'images' 
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def plot_training_history(log_data):
    """Generates Loss and mAP curves."""
    epochs = [x['epoch'] for x in log_data]
    loss = [x['train_loss'] for x in log_data]
    map_score = [x['val_map_50_95'] for x in log_data]

    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(epochs, loss, marker='o', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_ylabel('Loss')
    
    ax2.plot(epochs, map_score, marker='o', color='orange', label='Val mAP 0.5:0.95')
    ax2.set_title('Validation mAP')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_results.png')
    print(f"Saved plots to {OUTPUT_DIR / 'training_results.png'}")

def visualize_predictions(model_path, img_path):
    """Draws boxes on a test image."""
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    img = Image.open(img_path).convert("RGB")
    img_tensor = F.to_tensor(img).to(DEVICE)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    draw = ImageDraw.Draw(img)
    
    # Filter low confidence
    boxes = prediction['boxes'][prediction['scores'] > 0.5]
    labels = prediction['labels'][prediction['scores'] > 0.5]

    for box, label in zip(boxes, labels):
        color = "red" if label == 1 else "green" # 1 is Anomaly
        draw.rectangle(box.cpu().numpy(), outline=color, width=3)

    save_path = OUTPUT_DIR / f"pred_{os.path.basename(img_path)}"
    img.save(save_path)
    print(f"Saved prediction to {save_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Plot Graphs
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            data = json.load(f)
        plot_training_history(data)
        
        # Find best model
        best_epoch = max(data, key=lambda x: x['val_map_50_95'])
        best_model_path = MODEL_DIR / f"model_epoch_{best_epoch['epoch']}.pth"
        print(f"Best Model: Epoch {best_epoch['epoch']} (mAP: {best_epoch['val_map_50_95']})")

        # 2. Visualize Inference on first 3 test images
        test_images = list(TEST_IMG_DIR.glob('*.png'))[:3]
        for img in test_images:
            visualize_predictions(best_model_path, img)