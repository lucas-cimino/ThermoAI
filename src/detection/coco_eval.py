import os
import time
import torch
import torch.distributed as dist
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
import numpy as np

def to_list(x):
    """Converts a tensor or array to a Python list."""
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    elif isinstance(x, np.ndarray):
        x = x.tolist()
    return x

class CocoEvaluator:
    """
    COCO Evaluator class using pycocotools.COCOeval to compute metrics.
    """
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = defaultdict(list)

    def update(self, predictions):
        img_ids = list(predictions.keys())
        self.img_ids.extend(img_ids)
        for iou_type in self.iou_types:
            results = self.prepare_for_coco_eval(predictions, iou_type)
            self.eval_imgs[iou_type].append(results)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 0)
        
        create_and_update_coco_eval(self.coco_eval, self.img_ids, self.eval_imgs)

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("Summary for {}".format(iou_type))
            coco_eval.summarize()

    def prepare_for_coco_eval(self, predictions, iou_type):
        """
        Prepares detection results in COCO format (list of dicts).
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = to_list(boxes.cpu())
            scores = to_list(prediction["scores"].cpu())
            labels = to_list(prediction["labels"].cpu())

            if iou_type == "bbox":
                for box, score, label in zip(boxes, scores, labels):
                    # COCO needs [x_min, y_min, width, height] format
                    box[2] = box[2] - box[0] 
                    box[3] = box[3] - box[1] 
                    coco_results.append({
                        "image_id": original_id,
                        "category_id": label,
                        "bbox": box,
                        "score": score,
                    })
        return coco_results

def create_and_update_coco_eval(coco_eval, img_ids, eval_imgs):
    """
    Helper function to load results into COCOeval object for accumulation.
    """
    for iou_type, coco_evaluator in coco_eval.items():
        if iou_type == "bbox":
            result_json = list(eval_imgs[iou_type])
            
            # Create a mock COCO result object for initialization
            coco_dt = coco_evaluator.cocoGt.loadRes(result_json)
            
            # Assign the detection results to the COCOeval object
            coco_evaluator.cocoDt = coco_dt
            coco_evaluator.params.imgIds = img_ids