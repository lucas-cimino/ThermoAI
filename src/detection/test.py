import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.detection.coco_utils import CocoDetection
from src.detection.utils import collate_fn
from src.detection.engine import evaluate

# Config
TEST_DIR = BASE_DIR / 'data' / 'test'
TEST_JSON = TEST_DIR / 'test.json'
TEST_IMG_DIR = TEST_DIR / 'images'
# Change this to your best model path manually or via arg
BEST_MODEL_PATH = BASE_DIR / 'models' / 'model_epoch_25.pth' 
NUM_CLASSES = 2

def get_transform():
    return T.Compose([T.PILToTensor(), T.ToDtype(torch.float, scale=True)])

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Testing on {device}")

    # Load Test Data
    dataset_test = CocoDetection(str(TEST_IMG_DIR), str(TEST_JSON), get_transform())
    loader_test = DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Load Model
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)

    # Run Evaluation
    print("Running evaluation on TEST set...")
    coco_eval = evaluate(model, loader_test, device=device)
    
    stats = coco_eval.coco_eval['bbox'].stats
    # Score = (mAP50-95 * 0.8) + (Precision * 0.1) + (Recall * 0.1)
    # Mapping: mAP50-95=stats[0], Precision=stats[1] (approx mAP50), Recall=stats[8]
    final_score = (stats[0] * 0.8) + (stats[1] * 0.1) + (stats[8] * 0.1)
    
    print("\n========================================")
    print(f"FINAL TEST SCORE: {final_score:.4f}")
    print("========================================")

if __name__ == "__main__":
    main()