from PIL import Image
import os
import json
import torch
from torchvision.datasets import VisionDataset
from pycocotools.coco import COCO

class CocoDetection(VisionDataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(None, transforms=transforms) 
        self.img_folder = img_folder
        self.transforms = transforms
        
        self.coco = COCO(ann_file) 
        self.ids = sorted(self.coco.getImgIds()) 

    def __getitem__(self, idx):
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
            x, y, w, h = ann['bbox']
            
            if w > 0 and h > 0:
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
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
    
def get_coco_api_from_dataset(dataset):
    """
    Retrieves the COCO API object from a dataset, used for evaluation.
    """
    return dataset.coco