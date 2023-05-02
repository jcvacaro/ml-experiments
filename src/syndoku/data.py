from copy import deepcopy
import torch
from torch.utils.data import DataLoader, Subset

from dataset import SyndokuDataset
import transforms as T

def collate_fn(batch):
    return tuple(zip(*batch))

class SyndokuData:
    def __init__(self, args):
        self.args = args
        self.dataset = SyndokuDataset(self.args.dataset_dir)

        # select a training subset if specified
        indices = torch.randperm(len(self.dataset)).tolist()
        if self.args.dataset_train_subset > 0:
            indices = indices[:self.args.dataset_train_subset]

        # train/val dataset split
        val_size = int(self.args.dataset_val_split * len(indices))
        self.dt_train = Subset(self.dataset, indices[:-val_size])
        self.dt_val = Subset(self.dataset, indices[-val_size:])

        # data loaders
        self.dt_train_loader = DataLoader(dataset=self.dt_train, batch_size=self.args.batch_size, num_workers=self.args.workers, collate_fn=collate_fn)
        self.dt_train_loader.dataset.dataset = deepcopy(self.dataset)
        self.dt_train_loader.dataset.dataset.transforms = self.train_transform()
        self.dt_val_loader = DataLoader(dataset=self.dt_val, batch_size=1, num_workers=self.args.workers, collate_fn=collate_fn)
        self.dt_val_loader.dataset.dataset.transforms = self.val_transform()

        print('training dataset size:', len(self.dt_train))
        print('Validation dataset size:', len(self.dt_val))
        print('Dataset labels:', self.dataset.num_classes)

    @property
    def num_classes(self):
        return self.dataset.num_classes

    @property
    def train_transform(self):
        return T.Compose([
            T.Resize(448, 448), 
            T.RandomChoice([
                T.RandomAdjustSharpness(0.7),
                T.RandomAutocontrast(),
                T.RandomEqualize(),
                T.RandomInvert(),
            ]),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
        ])

    @property
    def val_transform(self):
        return T.Compose([
            T.Resize(448, 448), 
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
        ])

    @property
    def train_dataset(self):
        return self.dt_train

    @property
    def val_dataset(self):
        return self.dt_val

    @property
    def train_dataloader(self):
        return self.dt_train_loader

    @property
    def val_dataloader(self):
        return self.dt_val_loader

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
