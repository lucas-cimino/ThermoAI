# step_5_train_rtdetr.py
# Train RT-DETR (ResNet-based) as an accuracy-push line.
# Uses same data.yaml (2 classes).

from ultralytics import YOLO

# ---- Paths & knobs ----
MODEL_WEIGHTS = "rtdetr-l.pt"   # try 'rtdetr-x.pt' or backbone variants if available
DATA_YAML     = "data.yaml"
PROJECT       = "runs_thermoai"
RUN_NAME      = "rtdetr_l_2cls"
DEVICE        = 0

if __name__ == "__main__":
    model = YOLO(MODEL_WEIGHTS)

    results = model.train(
        data=DATA_YAML,
        imgsz=640,
        epochs=300,
        batch=12,            # RT-DETR typically needs a slightly smaller batch
        device=DEVICE,
        workers=4,
        project=PROJECT,
        name=RUN_NAME,
        # Keep color augs off; modest geometric augs
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        flipud=0.0, fliplr=0.5,
        degrees=10, translate=0.05, scale=0.2, shear=0.0,
        mosaic=0.5, mixup=0.1,
        patience=75,
        save=True,
    )

    print("RT-DETR training finished. Best weights:",
          f"{results.save_dir}/weights/best.pt")
