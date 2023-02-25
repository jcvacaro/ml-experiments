import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from dataset import SyndokuDataset
import transforms

def collate_fn(batch):
    return tuple(zip(*batch))

class SyndokuData:
    def __init__(self, args):
        self.args = args
        self.setup(None)

    @property
    def num_classes(self):
        return self.dataset.num_classes

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize(448, 448), 
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ])

    @property
    def val_transform(self):
        return self.train_transform

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.dataset = SyndokuDataset(self.args.dataset_dir, self.val_transform)
            # select a training subset if specified
            indices = torch.randperm(len(self.dataset)).tolist()
            if self.args.dataset_train_subset > 0:
                indices = indices[:self.args.dataset_train_subset]
            # train/val dataset split
            val_size = int(self.args.dataset_val_split * len(indices))
            self.dt_train = Subset(self.dataset, indices[:-val_size])
            self.dt_val = Subset(self.dataset, indices[-val_size:])
            self.dims = self.dt_train[0][0].shape
            print('training dataset size:', len(self.dt_train))
            print('Validation dataset size:', len(self.dt_val))
            print('Dataset labels:', self.dataset.num_classes)

    def train_dataset(self):
        return self.dt_train

    def val_dataset(self):
        return self.dt_val

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset(), batch_size=self.args.batch_size, num_workers=self.args.workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset(), batch_size=1, num_workers=self.args.workers, collate_fn=collate_fn)

if __name__ == '__main__':
    from argparse import ArgumentParser
    import os
    from pathlib import Path
    import train

    parser = ArgumentParser()
    parser = train.add_argparse_args(parser)
    args = parser.parse_args('')
    args.dataset_dir = Path.home() / 'data/syndoku'
    data = SyndokuData(args)
    dataset = data.train_dataset().dataset
    print('dataset len:', len(dataset))

    errors = 0
    for i in range(len(dataset)):
        try:
            img, target = dataset[i]
            boxes = target['boxes']
            if (boxes[:, :] <= 0).any() or (boxes[:, :] >= 448).any():
                raise ValueError()
        except:
            errors += 1
            print(dataset.get_image_path(i))
            print(dataset.get_label_path(i))

    print('errors:', errors)
