import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import SyndokuDataset

def collate_fn(batch):
    return tuple(zip(*batch))

class SyndokuData:
    def __init__(self, args):
        self.args = args
        self.setup(None)

    @property
    def num_classes(self):
        return SyndokuDataset.num_classes

    @property
    def train_transform(self):
        return A.Compose([
            A.Resize(256, 256), 
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    @property
    def val_transform(self):
        return self.train_transform

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.dataset = SyndokuDataset(self.args.dataset_dir, self.val_transform)
            # select a training subset if specified
            indices = torch.randperm(len(self.dataset)).tolist()
            if self.args.dataset_train_subset > 0:
                indices = indices[:self.args.train_dataset_subset]
            # train/val dataset split
            val_size = int(self.args.dataset_val_split * len(indices))
            self.dt_train = Subset(self.dataset, indices[:-val_size])
            self.dt_val = Subset(self.dataset, indices[-val_size:])
            self.dims = self.dt_train[0][0].shape

    def train_dataset(self):
        return self.dt_train

    def val_dataset(self):
        # return self.dt_val
        return self.dt_train

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset(), batch_size=self.args.batch_size, num_workers=self.args.workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset(), batch_size=1, num_workers=self.args.workers, collate_fn=collate_fn)
