import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.utils import draw_bounding_boxes, save_image
from PIL import Image
import os
import time

# --- Configuration ---
NUM_CLASSES = 2  # Must match training: 1 (anomaly) + 1 (background)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL_PATH = './models/faster_rcnn_final_epoch_25.pth'
TEST_IMAGE_PATH = './test_image.jpg' # IMPORTANT: Place a test image here
OUTPUT_DIR = './predictions'
THRESHOLD = 0.7 # Only show predictions with confidence > 70%

def get_model(num_classes):
    """Initializes the Faster R-CNN model structure."""
    model = fasterrcnn_resnet50_fpn_v2(weights=None) # Start without pre-trained weights
    
    # Replace the box predictor head (using the corrected FastRCNNPredictor)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def run_inference():
    # 0. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Trained model not found at {MODEL_PATH}. Did training finish?")
        return
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"ERROR: Test image not found at {TEST_IMAGE_PATH}. Please place an image there.")
        return

    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Set to evaluation mode
    
    # 2. Load Image
    print(f"Processing image: {TEST_IMAGE_PATH}")
    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    
    # Convert PIL image to Tensor (C, H, W)
    img_tensor = F.to_tensor(image).to(DEVICE)
    
    # 3. Run Inference
    with torch.no_grad():
        # The model expects a list of tensors, even for a single image
        predictions = model([img_tensor]) 

    # 4. Process and Visualize Results
    pred = predictions[0]
    
    # Filter predictions by confidence threshold
    high_confidence_mask = pred['scores'] > THRESHOLD
    boxes = pred['boxes'][high_confidence_mask]
    scores = pred['scores'][high_confidence_mask]

    # Convert image tensor back to 8-bit format for drawing
    img_tensor_uint8 = (img_tensor * 255).byte()

    if boxes.shape[0] > 0:
        print(f"Found {boxes.shape[0]} anomalies with confidence > {THRESHOLD}.")
        # Draw the bounding boxes on the image tensor
        result_img_tensor = draw_bounding_boxes(
            image=img_tensor_uint8.cpu(), 
            boxes=boxes.cpu(), 
            colors="red", 
            width=3
        )
    else:
        print("No anomalies detected above the confidence threshold.")
        result_img_tensor = img_tensor_uint8.cpu()

    # 5. Save Output
    output_filename = os.path.join(OUTPUT_DIR, f"prediction_{int(time.time())}.png")
    save_image(result_img_tensor.float() / 255, output_filename)
    print(f"Prediction image saved to: {output_filename}")


if __name__ == '__main__':
    run_inference()