import torch

# This function is crucial for Object Detection DataLoaders
# It correctly handles batches where images and targets have variable sizes.
def collate_fn(batch):
    """
    Collate function to handle variable sized inputs in a batch.
    It takes a list of (img, target) tuples and returns a tuple of lists 
    ([img1, img2, ...], [target1, target2, ...]).
    """
    return tuple(zip(*batch))