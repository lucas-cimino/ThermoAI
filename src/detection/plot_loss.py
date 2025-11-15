import matplotlib.pyplot as plt
import numpy as np
import os

# --- Loss Data from Your Training Log ---
# This data is manually copied from the output of your completed 25-epoch run.
# A real training script would save this data automatically to a log file.
LOSS_VALUES = [
    0.0738, 0.0471, 0.0395, 0.0318, 0.0292, 0.0254, 0.0231, 0.0214, 0.0187,
    0.0182, 0.0166, 0.0162, 0.0164, 0.0152, 0.0128, 0.0126, 0.0123, 0.0113,
    0.0115, 0.0108, 0.0103, 0.0101, 0.0100, 0.0092, 0.0091
]
NUM_EPOCHS = len(LOSS_VALUES)
OUTPUT_DIR = './plots'

def plot_training_loss():
    """Generates and saves a plot of the training loss over epochs."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create the x-axis labels (Epochs 1 to 25)
    epochs = np.arange(1, NUM_EPOCHS + 1)
    
    # Initialize the figure and axes
    plt.figure(figsize=(10, 6))
    
    # Plot the data
    plt.plot(epochs, LOSS_VALUES, marker='o', linestyle='-', color='#00aaff', label='Training Loss')
    
    # Add title and labels
    plt.title('Faster R-CNN Training Loss Curve (25 Epochs)', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    
    # Add grid lines for readability
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight the convergence with annotations
    plt.annotate(
        f'Initial Loss: {LOSS_VALUES[0]}', 
        (epochs[0], LOSS_VALUES[0]), 
        textcoords="offset points", 
        xytext=(-10, 15), 
        ha='center', 
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
    )
    plt.annotate(
        f'Final Loss: {LOSS_VALUES[-1]}', 
        (epochs[-1], LOSS_VALUES[-1]), 
        textcoords="offset points", 
        xytext=(10, -20), 
        ha='center', 
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2")
    )
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot to a file
    output_path = os.path.join(OUTPUT_DIR, 'training_loss_curve.png')
    plt.savefig(output_path)
    print(f"Loss curve visualization saved to: {output_path}")

if __name__ == '__main__':
    # Matplotlib needs 'Agg' backend in non-GUI environments like JupyterLab terminal
    plt.switch_backend('Agg') 
    plot_training_loss()