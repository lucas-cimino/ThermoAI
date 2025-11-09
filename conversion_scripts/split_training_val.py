import os
import random
import shutil

# Paths
IMG_DIR = "dataset/images/train"
LBL_DIR = "dataset/labels/train"

VAL_IMG_DIR = "dataset/images/val"
VAL_LBL_DIR = "dataset/labels/val"

SPLIT_RATIO = 0.2  # 20% validation

def ensure_dirs():
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LBL_DIR, exist_ok=True)

def main():
    ensure_dirs()

    images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(images)

    val_size = int(len(images) * SPLIT_RATIO)
    val_files = images[:val_size]

    for img in val_files:
        img_src = os.path.join(IMG_DIR, img)
        img_dst = os.path.join(VAL_IMG_DIR, img)
        shutil.move(img_src, img_dst)

        label_name = os.path.splitext(img)[0] + ".txt"
        lbl_src = os.path.join(LBL_DIR, label_name)
        lbl_dst = os.path.join(VAL_LBL_DIR, label_name)

        if os.path.exists(lbl_src):
            shutil.move(lbl_src, lbl_dst)
        else:
            print(f"[WARNING] Label missing for: {img}")

    print(f"✅ Split Complete!")
    print(f"Train images remaining: {len(os.listdir(IMG_DIR))}")
    print(f"Val images moved: {len(os.listdir(VAL_IMG_DIR))}")

if __name__ == "__main__":
    main()
