import matplotlib.pyplot as plt
import numpy as np
import os

# --- Data Simulation ---
# 1. Training Loss (your actual data, smoothed for a cleaner plot)
TRAIN_LOSS = [
    0.0738, 0.0471, 0.0395, 0.0318, 0.0292, 0.0254, 0.0231, 0.0214, 0.0187,
    0.0182, 0.0166, 0.0162, 0.0164, 0.0152, 0.0128, 0.0126, 0.0123, 0.0113,
    0.0115, 0.0108, 0.0103, 0.0101, 0.0100, 0.0092, 0.0091
]
NUM_EPOCHS = len(TRAIN_LOSS)
EPOCHS = np.arange(1, NUM_EPOCHS + 1)

# 2. Simulated mAP/Accuracy Data (Replace with real data after running evaluate.py)
# Note: In object detection, AP@0.50 is often used as the accuracy surrogate.
# We'll simulate increasing performance mirroring the loss drop.
AP50_TRAIN = [0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.53, 0.54, 0.55, 
              0.56, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64,
              0.65, 0.65, 0.66, 0.67, 0.67] # mAP@0.50 on Training Data
AP50_VAL = [0.25, 0.30, 0.34, 0.38, 0.41, 0.43, 0.44, 0.45, 0.46, 0.47,
            0.48, 0.48, 0.49, 0.49, 0.50, 0.50, 0.51, 0.51, 0.51, 0.52,
            0.52, 0.52, 0.53, 0.53, 0.54] # mAP@0.50 on Validation Data

OUTPUT_DIR = './plots'

def plot_advanced_metrics():
    """Generates a multi-panel plot of training metrics."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Setup the figure with two subplots (side-by-side or stacked)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('Faster R-CNN Training and Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # --- Top Plot: Training Loss ---
    ax1 = axes[0]
    ax1.plot(EPOCHS, TRAIN_LOSS, marker='o', linestyle='-', color='#00aaff', label='Total Training Loss')
    ax1.set_title('Training Loss Convergence', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- Bottom Plot: mAP / Accuracy ---
    ax2 = axes[1]
    ax2.plot(EPOCHS, AP50_TRAIN, marker='o', linestyle='-', color='teal', label='Training mAP@0.50')
    ax2.plot(EPOCHS, AP50_VAL, marker='o', linestyle='-', color='darkorange', label='Validation mAP@0.50')
    
    ax2.set_title('Model Performance (mAP@0.50)', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('mAP Score', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust for suptitle
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'advanced_metrics_plot.png')
    plt.savefig(output_path)
    print(f"Advanced metrics visualization saved to: {output_path}")

if __name__ == '__main__':
    plt.switch_backend('Agg') 
    plot_advanced_metrics()