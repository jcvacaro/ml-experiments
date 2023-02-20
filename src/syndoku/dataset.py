import glob
import json
import numpy as np
import os
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
from typing import Mapping

def document_iterate(element):
        if isinstance(element, list):
            for e in element:
                if isinstance(e, list) or isinstance(e, Mapping):
                    yield from document_iterate(e)
        elif isinstance(element, Mapping):
            for k,v in element.items():
                if isinstance(v, list) or isinstance(v, Mapping):
                    yield from document_iterate(v)
            if 'label' in element:
                yield element

class SyndokuDataset(Dataset):
    label_dict = {
        'token': 1,
        'image': 2,
        'bullet': 3,
    }
    num_classes = len(list(label_dict.keys()))

    def __init__(self, root_dir, transforms):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.files = sorted(glob.glob(os.path.join(self.root_dir, '*.png')))

    def get_classes(self):
        return list(self.label_dict.keys())

    def __len__(self):
        return len(self.files)

    def get_image_path(self, idx):
        return self.files[idx]

    def get_label_path(self, idx):
        return self.get_image_path(idx).replace('.png', '.json')

    def fix_degenerate_boxes(self, boxes):
        degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        if degenerate_boxes.any():
            boxes[degenerate_boxes[:, 0], 2:] +=  1

    def __getitem__(self, idx):
        # get the image
        img = Image.open(self.get_image_path(idx)).convert('RGB')

        # extract bounding boxes from the labels
        doc = json.load(open(self.get_label_path(idx)))
        bboxes = []
        labels = []
        for element in document_iterate(doc):
            if element['label'] in self.label_dict:
                bboxes.append((element['bbox']['x1'], element['bbox']['y1'], element['bbox']['x2'], element['bbox']['y2']))
                labels.append(self.label_dict[element['label']])

        # prepare the target dictionary
        labels = torch.as_tensor(labels, dtype=torch.int64)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)

        # Check for degenerate boxes
        self.fix_degenerate_boxes(bboxes)

        target = {
            'boxes': bboxes,
            'labels': labels,
        }
        # if self.transforms is not None:
        img, target = self.transforms(img, target)
        return img, target
