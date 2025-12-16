#!/bin/bash

# Configuration
CONFIG_FILE="configs/dfine/custom/dfine_hgnetv2_s_obj2coco_test.yml"

# List of checkpoints to evaluate
CHECKPOINTS=(
    "output/dfine_hgnetv2_s_obj2coco_custom/best_stg2.pth"
    "experiments_storage/seed_999/dfine_hgnetv2_s_obj2coco_custom/best_stg2.pth"
    "experiments_storage/seed_2023/dfine_hgnetv2_s_obj2coco_custom/best_stg2.pth"
)

# Temp file to store logs
LOG_FILE="eval_temp.log"

# Initialize variables
TOTAL_SCORE=0
COUNT=0

echo "========================================================"
echo "Starting Evaluation for ${#CHECKPOINTS[@]} models..."
echo "Formula: Score = (mAP_50_95 * 0.8) + (Precision * 0.1) + (Recall * 0.1)"
echo "========================================================"

for checkpoint in "${CHECKPOINTS[@]}"; do
    ((COUNT++))
    echo ""
    echo "[Run $COUNT] Evaluating: $checkpoint"
    
    # Run the python command and pipe output to both console and a log file
    python3 train.py -c "$CONFIG_FILE" --test-only -r "$checkpoint" > "$LOG_FILE" 2>&1
    
    # Extract Precision
    PRECISION=$(grep "Metrics:" "$LOG_FILE" | sed -n "s/.*'precision': \([0-9.]*\).*/\1/p")
    
    # Extract Recall
    RECALL=$(grep "Metrics:" "$LOG_FILE" | sed -n "s/.*'recall': \([0-9.]*\).*/\1/p")
    
    # Extract mAP 50-95 (finds the specific AP line and takes the last number)
    MAP=$(grep "Average Precision" "$LOG_FILE" | grep "IoU=0.50:0.95" | grep "area=\s*all" | awk '{print $NF}')

    # Check if metrics were found
    if [[ -z "$MAP" || -z "$PRECISION" || -z "$RECALL" ]]; then
        echo "Error: Could not parse metrics from log. Check if the run finished successfully."
        continue
    fi

    # Calculate Score
    RUN_SCORE=$(awk -v map="$MAP" -v p="$PRECISION" -v r="$RECALL" 'BEGIN {print (map * 0.8) + (p * 0.1) + (r * 0.1)}')

    # 4. Output results for this run
    echo "   -> mAP(50-95): $MAP"
    echo "   -> Precision:  $PRECISION"
    echo "   -> Recall:     $RECALL"
    echo "   -> RUN SCORE:  $RUN_SCORE"

    # Add to total
    TOTAL_SCORE=$(awk -v total="$TOTAL_SCORE" -v score="$RUN_SCORE" 'BEGIN {print total + score}')
done

# Calculate Average
if [ "$COUNT" -gt 0 ]; then
    AVG_SCORE=$(awk -v total="$TOTAL_SCORE" -v count="$COUNT" 'BEGIN {print total / count}')
    
    echo ""
    echo "========================================================"
    echo "FINAL RESULTS"
    echo "========================================================"
    echo "Evaluated Models: $COUNT"
    echo "Accumulated Score: $TOTAL_SCORE"
    echo "AVERAGE SCORE:     $AVG_SCORE"
    echo "========================================================"
else
    echo "No runs completed successfully."
fi

# Cleanup
rm -f "$LOG_FILE"