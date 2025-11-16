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
        # Do not modify coco_gt here, pass it directly
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(self.coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = defaultdict(list)

    def update(self, predictions):
        img_ids = list(predictions.keys())
        self.img_ids.extend(img_ids)
        for iou_type in self.iou_types:
            results = self.prepare_for_coco_eval(predictions, iou_type)
            # Use append for list of lists
            self.eval_imgs[iou_type].append(results)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            # Concatenate all results from all updates
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 0)
        
        # This function loads the detection results (cocoDt) into the evaluator
        create_and_update_coco_eval(self.coco_eval, self.img_ids, self.eval_imgs)

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            # --- THIS IS THE FIX ---
            # We must call .evaluate() before .accumulate()
            # .evaluate() populates the self.evalImgs dict
            # .accumulate() uses that dict to calculate stats
            print("Running COCOeval.evaluate()...")
            coco_eval.evaluate()
            print("Running COCOeval.accumulate()...")
            coco_eval.accumulate()
            # ----------------------

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"Summary for iouType: {iou_type}")
            coco_eval.summarize()

    def prepare_for_coco_eval(self, predictions, iou_type):
        """
        Prepares detection results in COCO format (list of dicts).
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = to_list(prediction["boxes"])
            scores = to_list(prediction["scores"])
            labels = to_list(prediction["labels"])

            if iou_type == "bbox":
                for box, score, label in zip(boxes, scores, labels):
                    # COCO needs [x_min, y_min, width, height] format
                    # Ensure box coordinates are valid floats
                    box = [float(b) for b in box]
                    box[2] = box[2] - box[0] # width
                    box[3] = box[3] - box[1] # height
                    coco_results.append({
                        "image_id": original_id,
                        "category_id": int(label), # Ensure category_id is int
                        "bbox": box,
                        "score": float(score), # Ensure score is float
                    })
        return coco_results

def create_and_update_coco_eval(coco_eval, img_ids, eval_imgs):
    """
    Helper function to load results into COCOeval object for accumulation.
    """
    for iou_type, coco_evaluator in coco_eval.items():
        if iou_type == "bbox":
            # Ensure results are in the correct list format
            result_json = [item for sublist in eval_imgs[iou_type] for item in sublist]
            
            try:
                # Use loadRes to create the cocoDt object
                coco_dt = coco_evaluator.cocoGt.loadRes(result_json)
                coco_evaluator.cocoDt = coco_dt
            except Exception as e:
                print(f"Error loading results into COCO API: {e}")
                print("Dumping first 5 detection results for debugging:")
                print(json.dumps(result_json[:5], indent=2))
                return

            # Set the image IDs to evaluate
            coco_evaluator.params.imgIds = sorted(list(set(img_ids)))