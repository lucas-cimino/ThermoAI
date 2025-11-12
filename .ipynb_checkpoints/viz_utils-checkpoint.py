# ---- viz helpers ----
import cv2
import numpy as np
import os

CLASS_MAP = {0: "normal", 1: "anomaly"}  # your mapping

def draw_boxes(image_path, label_path, predictions, output_path):
    # class-color mapping: 0 = normal (🟩), 1 = anomaly (🟥)
    colors = {0: (0, 255, 0), 1: (0, 0, 255)}
    class_names = {0: "normal", 1: "anomaly"}

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # ---- draw ground-truth (dotted) boxes ----
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.strip().split())
                cls = int(cls)
                color = colors.get(cls, (255, 255, 255))
                label = class_names.get(cls, f"class {cls}")
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                # draw dotted rectangle
                step = 5
                for i in range(x1, x2, step * 2):
                    cv2.line(img, (i, y1), (min(i + step, x2), y1), color, 2)
                    cv2.line(img, (i, y2), (min(i + step, x2), y2), color, 2)
                for j in range(y1, y2, step * 2):
                    cv2.line(img, (x1, j), (x1, min(j + step, y2)), color, 2)
                    cv2.line(img, (x2, j), (x2, min(j + step, y2)), color, 2)

                cv2.putText(img, f"GT {label}", (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ---- draw predictions (solid) boxes ----
    for box in predictions.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        color = colors.get(cls, (255, 255, 255))
        label = class_names.get(cls, f"class {cls}")
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} ({conf:.2f})", (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(output_path, img)

