from PIL import Image
import os
import json
import torch
from torchvision.datasets import VisionDataset
from pycocotools.coco import COCO # <--- IMPORT THE ACTUAL COCO API

# This class handles loading COCO JSON and images, and converting annotations to tensors.
class CocoDetection(VisionDataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(None, transforms=transforms) 
        self.img_folder = img_folder
        self.transforms = transforms
        
        # --- CRITICAL FIX ---
        # Initialize the COCO API object from the annotation file
        # self.coco is now a COCO object, not a dict
        self.coco = COCO(ann_file) 
        # Get sorted image IDs using the API method
        self.ids = sorted(self.coco.getImgIds()) 
        # --------------------

    def __getitem__(self, idx):
        # Use the COCO API to get image and annotation info
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann_info = self.coco.loadAnns(ann_ids)
        
        # Get image file name
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        
        # Load the image and convert to RGB
        img = Image.open(os.path.join(self.img_folder, file_name)).convert("RGB")
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in ann_info:
            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # Check for valid bounding box (width and height > 0)
            if w > 0 and h > 0:
                # Convert to PyTorch format: [x_min, y_min, x_max, y_max]
                boxes.append([x, y, x + w, y + h]) 
                labels.append(ann['category_id'])
                areas.append(ann['area'])
                iscrowd.append(ann['iscrowd'])
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            # Apply the transforms
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
    
# --- This function is now correct ---
# It will return the COCO API object stored in dataset.coco
def get_coco_api_from_dataset(dataset):
    """
    Retrieves the COCO API object from a dataset, used for evaluation.
    """
    return dataset.coco