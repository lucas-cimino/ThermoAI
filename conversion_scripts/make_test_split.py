# make_test_split.py
# Moves ~10% of current TRAIN to TEST, stratified by class.
# Layout assumed:
# dataset/
#   images/train, labels/train
#   images/val,   labels/val
#   images/test,  labels/test   (created here)

from pathlib import Path
import random
import shutil

# ---- config ----
ROOT = Path(".")
IMG_TRAIN = ROOT / "dataset/images/train"
LBL_TRAIN = ROOT / "dataset/labels/train"
IMG_TEST  = ROOT / "dataset/images/test"
LBL_TEST  = ROOT / "dataset/labels/test"
TEST_RATIO = 0.10        # 10% of TOTAL -> approx take 300 from your 2400-train
SEED = 42

# Optional: if your filenames encode a "scene" prefix, set a splitter to group by it
# e.g., lambda p: p.name.split("_")[0]  # group by first token
GROUP_KEY = None  # or a function: lambda p: p.name.split("_")[0]

random.seed(SEED)
IMG_TEST.mkdir(parents=True, exist_ok=True)
LBL_TEST.mkdir(parents=True, exist_ok=True)

# Helper: classify an image as "anomaly" if any bbox has class 1; else "normal"
def image_class(label_path: Path) -> str:
    if not label_path.exists():
        return "background"  # no boxes
    txt = label_path.read_text().strip()
    if not txt:
        return "background"
    # YOLO format: class xc yc w h
    for line in txt.splitlines():
        cls = line.split()[0]
        if cls == "1":
            return "anomaly"
    return "normal"

# collect train image/label pairs
pairs = []
for img_path in sorted(IMG_TRAIN.glob("*.*")):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    lbl_path = LBL_TRAIN / (img_path.stem + ".txt")
    cls = image_class(lbl_path)
    key = GROUP_KEY(img_path) if GROUP_KEY else None
    pairs.append((img_path, lbl_path, cls, key))

# stratify (optionally by group)
def stratified_sample(pairs, take_n):
    # Split into buckets by class (ignore background unless you want it in test)
    buckets = {"normal": [], "anomaly": []}
    for p in pairs:
        buckets[p[2]].append(p)
    # how many to take per class (proportional)
    total = len(pairs)
    want = {
        "normal": round(take_n * len(buckets["normal"]) / total),
        "anomaly": take_n - round(take_n * len(buckets["normal"]) / total)
    }
    chosen = []
    for cls in ("normal", "anomaly"):
        pool = buckets[cls]
        if GROUP_KEY:
            # group-aware sampling: pick groups until we reach the target
            groups = {}
            for item in pool:
                groups.setdefault(item[3], []).append(item)
            keys = list(groups.keys())
            random.shuffle(keys)
            count = 0
            for k in keys:
                if count >= want[cls]: break
                chunk = groups[k]
                chosen.extend(chunk)
                count += len(chunk)
        else:
            random.shuffle(pool)
            chosen.extend(pool[:want[cls]])
    # De-duplicate in case of overlap
    chosen_set = set((c[0], c[1]) for c in chosen)
    chosen = [c for c in pairs if (c[0], c[1]) in chosen_set]
    return chosen

train_count = len(pairs)
# Approximate target: 10% of total (you have 3000 total → 300 test). You currently have 2400 in train:
take_n = round(TEST_RATIO * (2400 + 600))  # ~300
take_n = min(take_n, train_count)

choice = stratified_sample(pairs, take_n)
choice_set = set((c[0], c[1]) for c in choice)

print(f"Moving {len(choice)} image/label pairs to TEST...")

moved = 0
for img_path, lbl_path, cls, _ in choice:
    # move image
    dest_img = IMG_TEST / img_path.name
    shutil.move(str(img_path), dest_img)
    # move label (if missing, create empty)
    dest_lbl = LBL_TEST / lbl_path.name
    if lbl_path.exists():
        shutil.move(str(lbl_path), dest_lbl)
    else:
        dest_lbl.write_text("")  # keep alignments
    moved += 1

print(f"Done. Moved {moved} pairs.")
print("Final counts:")
print(f"  Train images: {len(list(IMG_TRAIN.glob('*.*')))}")
print(f"  Val   images: {len(list((ROOT/'dataset/images/val').glob('*.*')))}")
print(f"  Test  images: {len(list(IMG_TEST.glob('*.*')))}")
