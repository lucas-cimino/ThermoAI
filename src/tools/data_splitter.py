import json
import os
import random

# --- Configuration ---
# INPUT_JSON_PATH is now located in the data/ directory
INPUT_JSON_PATH = './dataset/train/train.json' 
OUTPUT_DIR = './dataset/data_split' # Output directory is also in data/
SPLIT_RATIO = 0.8 
RANDOM_SEED = 42   
# ---------------------

def split_coco_dataset(input_path, output_dir, ratio, seed):
    """Reads a COCO JSON file, splits the images, and creates two new COCO JSON files."""
    print(f"Starting data split with ratio {ratio}...")
    
    # 1. Load the original COCO file
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_path}. Ensure it is in the 'dataset/' folder.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    # 2. Extract and shuffle image IDs
    images = data['images']
    image_ids = [img['id'] for img in images]
    random.shuffle(image_ids)
    
    # Determine split index
    train_size = int(len(image_ids) * ratio)
    train_ids = set(image_ids[:train_size])
    val_ids = set(image_ids[train_size:])

    print(f"Total images: {len(image_ids)}")
    print(f"Train images: {len(train_ids)} ({ratio*100}%)")
    print(f"Validation images: {len(val_ids)} ({(1-ratio)*100}%)")

    # 3. Filter annotations and images for each set
    train_data = {'info': data.get('info', {}), 'licenses': data.get('licenses', []), 'categories': data['categories'], 'images': [], 'annotations': []}
    val_data = {'info': data.get('info', {}), 'licenses': data.get('licenses', []), 'categories': data['categories'], 'images': [], 'annotations': []}

    # Populate image lists
    id_to_image = {img['id']: img for img in images}
    train_data['images'] = [id_to_image[id] for id in train_ids]
    val_data['images'] = [id_to_image[id] for id in val_ids]

    # Populate annotation lists
    for ann in data['annotations']:
        if ann['image_id'] in train_ids:
            train_data['annotations'].append(ann)
        elif ann['image_id'] in val_ids:
            val_data['annotations'].append(ann)
    
    # 4. Save the new JSON files
    with open(os.path.join(output_dir, 'train_split.json'), 'w') as f:
        json.dump(train_data, f)
    print(f"Saved train data to {os.path.join(output_dir, 'train_split.json')}")

    with open(os.path.join(output_dir, 'val_split.json'), 'w') as f:
        json.dump(val_data, f)
    print(f"Saved validation data to {os.path.join(output_dir, 'val_split.json')}")

if __name__ == '__main__':
    # We run this script from the project root directory
    split_coco_dataset(INPUT_JSON_PATH, OUTPUT_DIR, SPLIT_RATIO, RANDOM_SEED)