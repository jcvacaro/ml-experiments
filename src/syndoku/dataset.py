import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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
        self.files = sorted(os.listdir(self.root_dir))

    def get_classes(self):
        return list(self.label_dict.keys())

    def __len__(self):
        return int(len(self.files) / 2)

    def get_image_path(self, idx):
        return os.path.join(self.root_dir, self.files[(idx * 2) + 1])

    def get_label_path(self, idx):
        return os.path.join(self.root_dir, self.files[idx * 2])

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
            if element['label'] in ['token', 'image']:
                bboxes.append((element['bbox']['x1'], element['bbox']['y1'], element['bbox']['x2'], element['bbox']['y2']))
                labels.append(self.label_dict[element['label']])

        # prepare the target dictionary
        image_id = torch.tensor([idx])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.zeros_like(labels) # suppose all instances are not crowd
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        if len(bboxes) <= 0:
            area = torch.zeros_like(bboxes)
        else:
            area = bboxes[:, 2] * bboxes[:, 3] # w*h

        # Check for degenerate boxes
        self.fix_degenerate_boxes(bboxes)

        target = {
            'boxes': bboxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd,
        }

        return self.transforms(img), target

if __name__ == '__main__':
    from pprint import pprint
    dt = SyndokuDataset('/data')
    print(f'size: {len(dt)}')
    print(f'classes: {dt.get_classes()}')
    print(f'num_classes: {dt.num_classes}')

    for i in range(3):
        img, tgt = dt[i]
        print(tgt)

    print('done')
