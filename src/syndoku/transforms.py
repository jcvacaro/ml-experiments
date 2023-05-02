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

class RandomChoice:
    """Apply single transformation randomly picked from a list. This transform does not support torchscript."""

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, *args):
        t = random.choices(self.transforms)
        return t(*args)

class Resize(nn.Module):
    def __init__(self, h, w, interp=None):
        self.new_h = h
        self.new_w = w
        if interp is None:
            interp = Image.BILINEAR
        self.interp = interp

    def forward(self, image, target):
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

class RandomSolarize(torch.nn.Module):
    """Solarize the image randomly with a given probability by inverting all pixel
    values above a threshold. If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        threshold (float): all pixels equal or above this value are inverted.
        p (float): probability of the image being solarized. Default value is 0.5
    """

    def __init__(self, threshold, p=0.5):
        super().__init__()
        self.threshold = threshold
        self.p = p

    def forward(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Image to be solarized.

        Returns:
            PIL Image or Tensor: Randomly solarized image.
        """
        if torch.rand(1).item() < self.p:
            return F.solarize(img, self.threshold), target
        return img, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold},p={self.p})"

class RandomAdjustSharpness(torch.nn.Module):
    """Adjust the sharpness of the image randomly with a given probability. If the image is torch Tensor,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness_factor (float):  How much to adjust the sharpness. Can be
            any non negative number. 0 gives a blurred image, 1 gives the
            original image while 2 increases the sharpness by a factor of 2.
        p (float): probability of the image being sharpened. Default value is 0.5
    """

    def __init__(self, sharpness_factor, p=0.5):
        super().__init__()
        self.sharpness_factor = sharpness_factor
        self.p = p

    def forward(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Image to be sharpened.

        Returns:
            PIL Image or Tensor: Randomly sharpened image.
        """
        if torch.rand(1).item() < self.p:
            return F.adjust_sharpness(img, self.sharpness_factor), target
        return img, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sharpness_factor={self.sharpness_factor},p={self.p})"

class RandomAutocontrast(torch.nn.Module):
    """Autocontrast the pixels of the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        p (float): probability of the image being autocontrasted. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Image to be autocontrasted.

        Returns:
            PIL Image or Tensor: Randomly autocontrasted image.
        """
        if torch.rand(1).item() < self.p:
            return F.autocontrast(img), target
        return img, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

class RandomEqualize(torch.nn.Module):
    """Equalize the histogram of the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "P", "L" or "RGB".

    Args:
        p (float): probability of the image being equalized. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Image to be equalized.

        Returns:
            PIL Image or Tensor: Randomly equalized image.
        """
        if torch.rand(1).item() < self.p:
            return F.equalize(img), target
        return img, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

class RandomInvert(torch.nn.Module):
    """Inverts the colors of the given image randomly with a given probability.
    If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        p (float): probability of the image being color inverted. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Image to be inverted.

        Returns:
            PIL Image or Tensor: Randomly color inverted image.
        """
        if torch.rand(1).item() < self.p:
            return F.invert(img), target
        return img, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
