import json
import os
import random
from pathlib import Path

def split_data(input_json_path, output_dir, split_ratio=0.8, seed=42):
    """Splits a COCO JSON into train and validation sets."""
    random.seed(seed)
    
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    # Shuffle images
    random.shuffle(images)
    
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    train_img_ids = {img['id'] for img in train_images}
    val_img_ids = {img['id'] for img in val_images}

    train_anns = [ann for ann in annotations if ann['image_id'] in train_img_ids]
    val_anns = [ann for ann in annotations if ann['image_id'] in val_img_ids]

    train_data = {'images': train_images, 'annotations': train_anns, 'categories': categories}
    val_data = {'images': val_images, 'annotations': val_anns, 'categories': categories}

    os.makedirs(output_dir, exist_ok=True)
    
    train_out = os.path.join(output_dir, 'train_split.json')
    val_out = os.path.join(output_dir, 'val_split.json')

    with open(train_out, 'w') as f:
        json.dump(train_data, f)
    with open(val_out, 'w') as f:
        json.dump(val_data, f)

    print(f"Split complete. Train: {len(train_images)}, Val: {len(val_images)}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    INPUT_FILE = BASE_DIR / 'data' / 'train' / 'train.json'
    OUTPUT_DIR = BASE_DIR / 'data' / 'train'
    
    split_data(INPUT_FILE, OUTPUT_DIR)