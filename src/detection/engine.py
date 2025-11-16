import torch
import sys
import math
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
from utils import MetricLogger, SmoothedValue, reduce_dict, warmup_lr_scheduler

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
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch+1)
    
    # Mocking the iterator structure for logging
    class MockIterator:
        def __init__(self, data_loader, epoch):
            self.total = len(data_loader)
            self.current = 0
            self.epoch = epoch
            self.loss_hist = AverageMeter()
            self.lr = optimizer.param_groups[0]["lr"]
        
    iterator = MockIterator(data_loader, epoch)
    total_epochs = 25 # Must match the number in faster_rcnn_trainer.py

    # Apply learning rate warmup for the first 500 iterations
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in data_loader:
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
        iterator.loss_hist.update(loss_value)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Backpropagation step
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if epoch == 0 and iterator.current < warmup_iters:
            lr_scheduler.step()
        
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
    Standard COCO evaluation loop, required to run mAP calculation.
    """
    n_threads = torch.get_num_threads()
    # torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

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
    # torch.set_num_threads(n_threads)
    return coco_evaluator