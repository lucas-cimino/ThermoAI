# step_4_eval_and_predict.py
# Evaluate the trained model and run predictions on val images.

from ultralytics import YOLO
from pathlib import Path

# ---- Config ----
PROJECT    = "runs_thermoai"
RUN_NAME   = "yolov8s_2cls"  # must match step_3 script
DATA_YAML  = "data.yaml"
IMGSZ      = 640
CONF_THRES = 0.25
VAL_DIR    = "dataset/images/val"

if __name__ == "__main__":
    best_pt = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    assert best_pt.exists(), f"Missing weights: {best_pt}"

    model = YOLO(str(best_pt))

    # Validation (mAP50-95, Precision, Recall)
    print("==> Running validation…")
    metrics = model.val(data=DATA_YAML, imgsz=IMGSZ, split="val")
    # metrics keys include: metrics.box.map, map50, precision, recall, etc.
    print({
        "mAP50-95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    })

    # Batch predictions over validation images (saved into runs/predict...)
    print("==> Running prediction/visualization…")
    preds = model.predict(
        source=VAL_DIR,
        imgsz=IMGSZ,
        conf=CONF_THRES,
        save=True
    )
    print("Predictions saved to:", preds[0].save_dir if preds else "n/a")
