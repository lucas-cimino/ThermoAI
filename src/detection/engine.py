import torch
import sys
import math
import time 
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
# Import the full utils, not just individual functions
from utils import MetricLogger, SmoothedValue, reduce_dict, warmup_lr_scheduler

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch + 1}]' 

    # Use a fixed total_epochs for display, matching the trainer
    total_epochs = 25 
    
    # Apply learning rate warmup for the first epoch
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # Use the MetricLogger's log_every to print progress
    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # Move images and targets to the GPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Get losses from the model
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        # Backpropagation step
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Update the logger
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # Final print for the epoch (now handled by log_every)
    print(f"\nTraining complete for epoch {epoch + 1}.")


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Standard COCO evaluation loop, required to run mAP calculation.
    """
    n_threads = torch.get_num_threads()
    # torch.set_num_threads(1) 
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"] # We are only evaluating bounding boxes
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        model_time = time.time()
        outputs = model(images)
        model_time = time.time() - model_time

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # Gather the results from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # Accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    # --- THIS IS THE FIX ---
    # Return the actual results object, not the wrapper
    return coco_evaluator.coco_eval['bbox']
    # ----------------------