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
from models.lstm.ffs_dataset import FacesFeaturesDataModule


class FacesFeaturesLSTM(nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=lstm_conf.num_hidden,
                 num_lstm_layers=lstm_conf.num_lstm_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=num_hidden,  # number of neurons in each layer
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_conf.dropout
        )
        self.classifier = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()  # for multi GPU purposes
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]
        return self.classifier(out)