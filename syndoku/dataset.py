import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from detectron2.data import transforms as T

class ICPR2020(Dataset):
    IMAGE_DIR = 'images'
    LABEL_DIR = 'annotations_JSON'
    CHART_TYPES = (
        'horizontal_bar',
    )
    LABEL_DICT = {
        'axis_title': 1,
        'chart_title': 2,
        'legend_label': 3,
        'legend_title': 4,
        'other': 5,
        'tick_grouping': 6,
        'tick_label': 7,
        'value_label': 8,
        'legend_patch': 9,
        'bar': 10,
    }
    num_classes = 10

    def __init__(self, root_dir, split='train', bbox_format='xyxy', transforms=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            bbox_format (string): 'xyxy' the default, or 'xywh'.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.bbox_format = bbox_format
        self.transforms = transforms
        self.label_paths = sorted(os.listdir(os.path.join(self.root_dir, self.split, self.LABEL_DIR, self.CHART_TYPES[0])))

    def get_classes(self):
        return list(self.LABEL_DICT.keys())

    def __len__(self):
        return len(self.label_paths)

    def get_image_path(self, idx):
        return os.path.join(self.root_dir, self.split, self.IMAGE_DIR, self.CHART_TYPES[0], self.label_paths[idx]).replace('.json', '.jpg')

    def get_label_path(self, idx):
        return os.path.join(self.root_dir, self.split, self.LABEL_DIR, self.CHART_TYPES[0], self.label_paths[idx])

    def find_by_id(self, _id, objs):
        for obj in objs:
            if obj['id'] == _id:
                return obj
        return None

    def fix_degenerate_boxes(self, boxes):
        degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        if degenerate_boxes.any():
            boxes[degenerate_boxes[:, 0], 2:] +=  1

    def __getitem__(self, idx):
        # get the image
        img = Image.open(self.get_image_path(idx)).convert('RGB')

        # extract bounding boxes from the labels
        in_obj = json.load(open(self.get_label_path(idx)))
        bboxes = []
        labels = []

        # task 1
        #chart_type = in_obj['task1']['output']['chart_type']

        # task 2 & 3
        for text_block in in_obj['task2']['output']['text_blocks']:
            bb = text_block['bb'] if 'bb' in text_block else text_block['polygon']
            _id = text_block['id']
            obj = self.find_by_id(_id, in_obj['task3']['output']['text_roles'])
            assert obj is not None
            X = (int(bb['x0']), int(bb['x1']), int(bb['x2']), int(bb['x3']))
            Y = (int(bb['y0']), int(bb['y1']), int(bb['y2']), int(bb['y3']))
            bboxes.append((np.min(X), np.min(Y), np.max(X), np.max(Y)))
            labels.append(self.LABEL_DICT[obj['role']])

        # task 5
        for obj in in_obj['task5']['output']['legend_pairs']:
            patch_bb = obj['bb']
            p1 = (int(patch_bb['x0']), int(patch_bb['y0']))
            p2 = (int(patch_bb['x0'] + patch_bb['width']), int(patch_bb['y0'] + patch_bb['height']))
            bboxes.append((p1[0], p1[1], p2[0], p2[1]))
            labels.append(self.LABEL_DICT['legend_patch'])

        # task 6
        for bb in in_obj['task6']['output']['visual elements']['bars']:
            p1 = (int(bb['x0']), int(bb['y0']))
            p2 = (int(bb['x0'] + bb['width']), int(bb['y0'] + bb['height']))
            bboxes.append((p1[0], p1[1], p2[0], p2[1]))
            labels.append(self.LABEL_DICT['bar'])

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

        if self.transforms is not None:
            input = T.AugInput(np.array(img), boxes=target['boxes'])
            transform = self.transforms(input)  # type: T.Transform
            img = input.image.copy()
            target['boxes'] = torch.as_tensor(input.boxes, dtype=torch.float32)

        return F.to_tensor(img), target

if __name__ == '__main__':
    dt = ICPR2020('/data/', split='train')
    print(f'size: {len(dt)}')
    print(f'classes: {dt.get_classes()}')
    for i in range(len(dt)):
        try:
            img, tgt = dt[i]
        except:
            path = dt.get_label_path(i)
            print(f'removing json file {path} at index {i}')
            os.remove(path)

    print('done')