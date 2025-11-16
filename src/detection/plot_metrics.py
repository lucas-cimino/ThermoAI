import matplotlib.pyplot as plt
import numpy as np
import os
import json # <--- IMPORT JSON
from pathlib import Path

# --- Configuration ---
# Path to the log file created by the trainer
METRICS_LOG_PATH = './models/training_metrics.json'
OUTPUT_DIR = './plots'
# ---------------------

def plot_final_metrics():
    """Generates and saves a plot of the final training metrics from the JSON log."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- NEW: Load data from JSON ---
    if not Path(METRICS_LOG_PATH).exists():
        print(f"ERROR: Metrics log file not found at {METRICS_LOG_PATH}")
        print("Please run the main training script first!")
        return

    print(f"Loading metrics from {METRICS_LOG_PATH}...")
    with open(METRICS_LOG_PATH, 'r') as f:
        metrics_log = json.load(f)
    
    # Extract the data into lists
    epochs = [d['epoch'] for d in metrics_log]
    loss_scores = [d['loss'] for d in metrics_log]
    map_scores = [d['mAP'] for d in metrics_log]
    map_50_scores = [d['mAP_50'] for d in metrics_log]
    # --------------------------------
    
    # Create the figure with two plots, one on top of the other
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Faster R-CNN Training Results', fontsize=16, fontweight='bold')

    # --- Plot 1: Training Loss ---
    ax1.plot(epochs, loss_scores, marker='o', linestyle='-', color='#007acc', label='Average Training Loss')
    ax1.set_title('Training Loss per Epoch')
    ax1.set_ylabel('Loss (Average)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- Plot 2: mAP Score ---
    ax2.plot(epochs, map_scores, marker='s', linestyle='-', color='#d62728', label='Validation mAP@.50:.95')
    ax2.plot(epochs, map_50_scores, marker='^', linestyle='--', color='#2ca02c', label='Validation mAP@.50')
    
    # Find and mark the best score
    best_epoch = np.argmax(map_scores)
    best_score = map_scores[best_epoch]
    ax2.plot(best_epoch + 1, best_score, 'o', color='gold', markersize=15, label=f'Best mAP@.50:.95: {best_score:.4f} at Epoch {best_epoch+1}')
    
    ax2.set_title('Validation mAP Score per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP Score')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for the main title
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'final_training_metrics.png')
    plt.savefig(output_path)
    print(f"Final metrics plot saved to: {output_path}")

if __name__ == '__main__':
    # Use 'Agg' backend for non-GUI terminal environment
    plt.switch_backend('Agg') 
    plot_final_metrics()