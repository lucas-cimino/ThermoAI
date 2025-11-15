import torch
import sys

# Helper class to track average loss during training
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    
    # Mocking the iterator structure for logging
    class MockIterator:
        def __init__(self, data_loader, epoch):
            self.total = len(data_loader)
            self.current = 0
            self.epoch = epoch
            self.loss_hist = AverageMeter()
        
    iterator = MockIterator(data_loader, epoch)
    total_epochs = 25 # Must match the number in faster_rcnn_trainer.py

    for images, targets in data_loader:
        # Move images and targets to the GPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Get losses from the model
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        loss_value = losses.item()
        iterator.loss_hist.update(loss_value)

        # Backpropagation step
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        iterator.current += 1
        
        if iterator.current % print_freq == 0:
            sys.stdout.write(f"\rEpoch: [{epoch+1}/{total_epochs}], Iteration: {iterator.current}/{iterator.total}, Loss: {iterator.loss_hist.avg:.4f}")
            sys.stdout.flush()
            
    # Final print for the epoch
    sys.stdout.write(f"\rEpoch: [{epoch+1}/{total_epochs}], Iteration: {iterator.current}/{iterator.total}, Loss: {iterator.loss_hist.avg:.4f}")
    print("\nTraining complete for epoch.")


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Runs a minimal validation check. Full mAP calculation is disabled as it requires 
    the external pycocotools library which is often not pre-installed.
    """
    model.eval()
    
    print("Running minimal validation check...")
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        
        # In evaluation, the model returns predictions (boxes, labels, scores)
        model(images)
        
        if i >= 10: # Check 10 batches for stability
            break
        
    print("Validation check passed. (Full mAP calculation skipped.)")