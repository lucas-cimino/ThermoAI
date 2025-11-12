# step_5_test_yolov8.py
# Run evaluation on test set and visualize predictions vs. ground truth

from ultralytics import YOLO
from pathlib import Path
from viz_utils import draw_boxes  # ✅ import visualization helper
import os

PROJECT = Path("runs_thermoai")
RUN_NAME = "yolov8s_2cls"
DATA_YAML = "data_resolved.yaml"
IMGSZ = 640
CONF_THRES = 0.25
TEST_DIR = Path("dataset/images/test")
LABEL_DIR = Path("dataset/labels/test")

def find_best(project: Path, run_hint: str) -> Path:
    cand = project / run_hint / "weights" / "best.pt"
    if cand.exists():
        return cand
    runs = sorted([p for p in project.glob(f"{run_hint}*") if p.is_dir()],
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if runs:
        alt = runs[0] / "weights" / "best.pt"
        if alt.exists():
            return alt
    raise FileNotFoundError(f"No best.pt found under {project}/{run_hint}*")

if __name__ == "__main__":
    best_pt = find_best(PROJECT, RUN_NAME)
    print("Using weights:", best_pt)
    model = YOLO(str(best_pt))

    print("==> Running test evaluation...")
    metrics = model.val(data=DATA_YAML, imgsz=IMGSZ, split="test")
    print({
        "mAP50-95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    })

    print("==> Generating annotated test predictions...")
    results = model.predict(source=str(TEST_DIR), imgsz=IMGSZ, conf=CONF_THRES, save=False)
    output_dir = PROJECT / "test_viz"
    output_dir.mkdir(exist_ok=True)

    for r in results:
        img_path = Path(r.path)
        label_path = LABEL_DIR / (img_path.stem + ".txt")
        out_path = output_dir / img_path.name
        draw_boxes(str(img_path), str(label_path), r, str(out_path))

    print(f"Annotated test images saved to: {output_dir}")
