import os
import sys
import pandas as pd
import numpy as np
from jedi.api.refactoring import inline
from tqdm.auto import tqdm
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import seaborn as sns
from pylab import rcParams
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import accuracy
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CURRENT_FACES_FEATURES_DATA_X, CURRENT_FACES_FEATURES_DATA_Y, CURRENT_MIN_NUM_SMILE_FRAMES
from models._config import nns_config as nns_conf
from models._config import LSTM_config as lstm_conf


class FacesFeaturesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ffs, auth = self.data[idx]
        return dict(
            faces_features=torch.tensor(ffs.to_numpy()),
            authenticity=torch.tensor(auth).long()
        )


class FacesFeaturesDataModule(pl.LightningDataModule):
    def __init__(self, train_data, test_data, batch_size):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = FacesFeaturesDataset(self.train_data)
        self.test_dataset = FacesFeaturesDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=cpu_count())

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=cpu_count())
