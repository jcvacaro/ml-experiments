import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from dataset import SyndokuDataset
import vision.references.detection.transforms as T

def collate_fn(batch):
    return tuple(zip(*batch))

class SyndokuData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def num_classes(self):
        return SyndokuDataset.num_classes

    @property
    def train_transform(self):
        return T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
        ])

    @property
    def val_transform(self):
        return T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
        ])

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

    def train_dataloader(self):
        return DataLoader(dataset=self.dt_train, batch_size=self.args.batch_size, num_workers=self.args.workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.dt_val, batch_size=1, num_workers=self.args.workers, collate_fn=collate_fn)

    def train_dataset(self):
        return self.dt_train

    def val_dataset(self):
        return self.dt_val
