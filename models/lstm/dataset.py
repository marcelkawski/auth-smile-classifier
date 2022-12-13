import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class FacesFeaturesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ffs, auth = self.data[idx]
        return dict(
            faces_features=torch.Tensor(ffs.to_numpy()),
            authenticity=torch.tensor(auth).long()
        )


class FacesFeaturesDataModule(pl.LightningDataModule):
    def __init__(self, train_data, test_data, batch_size):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.test_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = FacesFeaturesDataset(self.train_data)
        self.test_dataset = FacesFeaturesDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
