# step_3_train_yolov8.py
# Train a 2-class thermal detector (normal, anomaly) with YOLOv8
# Requires: pip install ultralytics==8.3.50

from ultralytics import YOLO

# ---- Paths & knobs ----
MODEL_WEIGHTS = "yolov8s.pt"           # change to yolov8m.pt if you have headroom
DATA_YAML     = "data_resolved.yaml"            # must list: 0: normal, 1: anomaly
PROJECT       = "runs_thermoai"
RUN_NAME      = "yolov8s_2cls"
DEVICE        = 0                     # GPU id; use 'cpu' to force CPU

# ---- Train ----
if __name__ == "__main__":
    model = YOLO(MODEL_WEIGHTS)

    results = model.train(
        data=DATA_YAML,
        imgsz=640,
        epochs=40,
        batch=16,
        device=DEVICE,
        workers=4,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        # Thermal-friendly augs (no color jitter)
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        flipud=0.0, fliplr=0.5,
        degrees=10, translate=0.05, scale=0.2, shear=0.0,
        mosaic=0.7, mixup=0.1,
        patience=15,                # early stop if no improvement
        save=True,
    )

    print("Training finished. Best weights:",
          f"{results.save_dir}/weights/best.pt")
