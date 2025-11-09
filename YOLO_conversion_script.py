import json
from pathlib import Path
from PIL import Image

ann_path = Path("dataset/annotations/train.json")   # your file
img_root = Path("dataset")                          # base path
out_root = img_root                                 # we keep structure identical

# Output directories
(lbl_train := out_root / "labels/train").mkdir(parents=True, exist_ok=True)

with ann_path.open("r", encoding="utf-8") as f:
    coco = json.load(f)

images = {im["id"]: im for im in coco["images"]}
anns = {}
for a in coco["annotations"]:
    anns.setdefault(a["image_id"], []).append(a)

for img_id, img_info in images.items():
    img_path = img_root / img_info["file_name"]   # USE EXACT FILE_NAME FROM JSON
    img = Image.open(img_path)
    W, H = img.size

    label_lines = []
    for a in anns.get(img_id, []):
        cid = a["category_id"]
        x, y, w, h = a["bbox"]
        xc = (x + w/2) / W
        yc = (y + h/2) / H
        wn = w / W
        hn = h / H
        label_lines.append(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    out_label = lbl_train / (Path(img_info["file_name"]).with_suffix(".txt").name)
    out_label.write_text("\n".join(label_lines))

print("✅ Conversion complete. Labels saved under dataset/labels/train/")
