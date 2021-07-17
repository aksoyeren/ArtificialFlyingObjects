import torch
import torchvision

from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from typing import List, Tuple, Dict, Optional

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(nn.Module):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target

class Resize(object):
    def __init__(self, shape, **kwargs):
        self.shape = shape
        self.kwargs = kwargs
        
    def __call__(self, image, target):
        image = F.resize(image, self.shape, **self.kwargs)
        #target = F.resize(target, self.shape, **self.kwargs)
        
        old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
        new_boxes = boxes / old_dims  # percent coordinates

        #if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims
        return image, new_boxes
    
class SquarePad:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, target):
        _,w, h = image.shape
        
        top = self.width - w
        right = self.height - h
        padding = (0, 0, right, top)
        return F.pad(image, padding, 1, 'constant'), target