import torch
from torch.utils.data import DataLoader, Subset
from detectron2.data import transforms as T
import pytorch_lightning as pl

import guia_mlutils as G
import dataset

def collate_fn(batch):
    return tuple(zip(*batch))

class ICPR2020(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def train_transform(self):
        return G.transforms.create_augmentations(self.args)

    @property
    def val_transform(self):
        return None

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.dataset = dataset.ICPR2020(self.args.dataset_dir, split='train', transforms=self.train_transform)
            # select a training subset if specified
            indices = torch.randperm(len(self.dataset)).tolist()
            if self.args.dataset_train_subset > 0:
                indices = indices[:self.args.train_dataset_subset]
            # train/val dataset split
            val_size = int(self.args.dataset_val_split * len(indices))
            self.dt_train = Subset(self.dataset, indices[:-val_size])
            self.dt_val = Subset(self.dataset, indices[-val_size:])
            self.dims = self.dt_train[0][0].shape

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.dt_test = dataset.ICPR2020(self.args.dataset_dir, split='test')
            self.dims = getattr(self, 'dims', self.dt_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(dataset=self.dt_train, batch_size=self.args.batch_size, num_workers=self.args.workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.dt_val, batch_size=1, num_workers=self.args.workers, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(dataset=self.dt_test, batch_size=1, num_workers=self.args.workers, collate_fn=collate_fn)

    def train_dataset(self):
        return self.dt_train

    def val_dataset(self):
        return self.dt_val

    def test_dataset(self):
        return self.dt_test

    @property
    def num_classes(self):
        return dataset.ICPR2020.num_classes
