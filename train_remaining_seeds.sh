#!/bin/bash
RESULT_DIR="output/dfine_hgnetv2_s_obj2coco_custom"
mkdir -p experiments_storage

# Move Seed 42 if not already moved
if [ -d "$RESULT_DIR" ]; then mv "$RESULT_DIR" "experiments_storage/seed_42"; fi

SEEDS=(2023 999)
for seed in "${SEEDS[@]}"; do
    echo "Training Seed: $seed"
    sed -i "s/^seed: .*/seed: $seed/" configs/dfine/custom/dfine_hgnetv2_s_obj2coco_custom.yml

    # Train
    export model=s
    python3 train.py -c configs/dfine/custom/dfine_hgnetv2_s_obj2coco_custom.yml -t dfine_s_obj2coco.pth --use-amp

    # Test (Automatic Evaluation)
    python3 train.py -c configs/dfine/custom/dfine_hgnetv2_s_obj2coco_test.yml --test-only -r "$RESULT_DIR/best_stg1.pth" > "experiments_storage/log_seed_$seed.txt"

    # Move Results
    mv "$RESULT_DIR" "experiments_storage/seed_$seed"
done
echo "Done! Check logs in experiments_storage/"
