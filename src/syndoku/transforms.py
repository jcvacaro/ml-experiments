from PIL import Image
import torch
from torch import nn, Tensor
import torchvision
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
from typing import Dict, List, Optional, Tuple, Union

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Resize:
    def __init__(self, h, w, interp=None):
        self.new_h = h
        self.new_w = w
        if interp is None:
            interp = Image.BILINEAR
        self.interp = interp

    def __call__(self, image, target):
        h, w = image.height, image.width
        image = image.resize((self.new_w, self.new_h), self.interp)
        coords = target['boxes']
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / h)
        coords[:, 2] = coords[:, 2] * (self.new_w * 1.0 / w)
        coords[:, 3] = coords[:, 3] * (self.new_h * 1.0 / h)
        return image, target

class PILToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target

class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target
