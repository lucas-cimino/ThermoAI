# Creates a data_resolved.yaml with absolute paths for Ultralytics
# Run this on each machine before training.

from pathlib import Path
import yaml

# Project root = folder that contains this script (adjust if you prefer)
PROJECT_ROOT = Path(__file__).resolve().parent

# If your dataset lives elsewhere, change this:
DATASET_ROOT = PROJECT_ROOT / "dataset"

paths = {
    "train": (DATASET_ROOT / "images" / "train").as_posix(),
    "val":   (DATASET_ROOT / "images" / "val").as_posix(),
}

data = {
    # Using absolute paths avoids Ultralytics' global datasets_dir entirely
    "train": paths["train"],
    "val": paths["val"],
    # Class order must match your labels
    "names": {0: "normal", 1: "anomaly"},
}

OUT = PROJECT_ROOT / "data_resolved.yaml"
OUT.write_text(yaml.safe_dump(data, sort_keys=False))
print("Wrote:", OUT)
print("train ->", paths["train"])
print("val   ->", paths["val"])
