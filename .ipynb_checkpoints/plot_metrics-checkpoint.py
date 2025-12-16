import json
import matplotlib.pyplot as plt
import sys
import os

LOG_FILE = "output/dfine_hgnetv2_s_obj2coco_custom/log.txt"

def plot_logs(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    epochs = []
    losses = []
    map_50_95 = []
    lrs = []

    print(f"Reading {file_path}...")
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
                
                if 'epoch' in data:
                    epochs.append(data['epoch'])
                    losses.append(data.get('train_loss', 0))
                    lrs.append(data.get('train_lr', 0))
                    
                    if 'test_coco_eval_bbox' in data:
                        map_50_95.append(data['test_coco_eval_bbox'][0])
                    else:
                        map_50_95.append(map_50_95[-1] if map_50_95 else 0)
                        
            except json.JSONDecodeError:
                continue

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, losses, color='red', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, map_50_95, color='blue', label='mAP 50-95')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Accuracy')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, lrs, color='green', label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    output_image = "training_metrics_f.png"
    plt.savefig(output_image)
    print(f"Done! Plot saved to: {output_image}")

if __name__ == "__main__":
    plot_logs(LOG_FILE)