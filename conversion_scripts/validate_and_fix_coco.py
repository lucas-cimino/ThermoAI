import json
from pathlib import Path

ANNOT_PATH = Path("../dataset/annotations/train.json")
IMAGE_DIR = Path("../dataset/images/train")

with open(ANNOT_PATH, "r") as f:
    data = json.load(f)

images = data.get("images", [])
annotations = data.get("annotations", [])
categories = data.get("categories", [])

print(f"Loaded: {len(images)} images, {len(annotations)} annotations, {len(categories)} categories")

# --------------------
# 1) Ensure categories are 0-based and continuous
# --------------------
categories_sorted = sorted(categories, key=lambda x: x["id"])
id_map = {cat["id"]: i for i, cat in enumerate(categories_sorted)}  # old → new ID mapping

for cat in categories_sorted:
    cat["id"] = id_map[cat["id"]]

# Update annotation category_ids
for ann in annotations:
    ann["category_id"] = id_map[ann["category_id"]]

# --------------------
# 2) Remove annotations that reference non-existing images
# --------------------
image_ids = {img["id"] for img in images}
annotations = [ann for ann in annotations if ann["image_id"] in image_ids]

# --------------------
# 3) Remove image entries that do not actually exist on disk
# --------------------
valid_images = []
for img in images:
    img_path = IMAGE_DIR / img["file_name"]
    if img_path.exists():
        valid_images.append(img)

images = valid_images

# --------------------
# 4) Write cleaned file back
# --------------------
data["images"] = images
data["annotations"] = annotations
data["categories"] = categories_sorted

with open("../dataset/annotations/train_fixed.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ train_fixed.json created successfully!")
print(f"Images kept: {len(images)} | Annotations kept: {len(annotations)} | Categories: {len(categories_sorted)}")
