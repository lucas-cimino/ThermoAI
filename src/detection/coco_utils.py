from PIL import Image
import os
import json
import torch
from torchvision.datasets import VisionDataset

# This class handles loading COCO JSON and images, and converting annotations to tensors.
class CocoDetection(VisionDataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        # We pass None for root as we handle img_folder manually based on our project structure
        super().__init__(None, transforms=transforms) 
        self.img_folder = img_folder
        
        # Load the JSON file
        with open(ann_file, 'r') as f:
            coco = json.load(f)
        
        self.ids = [img['id'] for img in coco['images']]
        self.coco = coco
        
        # Create map from image ID to image info and annotations
        self.img_id_to_img = {img['id']: img for img in coco['images']}
        self.img_id_to_anns = {img_id: [] for img_id in self.ids}
        for ann in coco['annotations']:
            if ann['image_id'] in self.img_id_to_anns:
                self.img_id_to_anns[ann['image_id']].append(ann)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.img_id_to_img[img_id]
        ann_info = self.img_id_to_anns[img_id]
        
        file_name = img_info['file_name']
        
        # Load the image and convert to RGB
        img = Image.open(os.path.join(self.img_folder, file_name)).convert("RGB")
        
        boxes = []
        labels = []
        
        for ann in ann_info:
            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to PyTorch format: [x_min, y_min, x_max, y_max]
            boxes.append([x, y, x + w, y + h]) 
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])

        if self.transforms is not None:
            # Apply the transforms defined in faster_rcnn_trainer.py
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
    
    # Required for the evaluation stub in engine.py
    def get_coco_api_from_dataset(self):
        return self.coco