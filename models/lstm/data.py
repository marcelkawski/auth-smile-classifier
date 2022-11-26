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
from tensorboard import program
from torchmetrics.functional import accuracy
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CURRENT_FACES_FEATURES_DATA_X, CURRENT_FACES_FEATURES_DATA_Y
from models._config import nns_config as nns_conf
from models._config import LSTM_config as lstm_conf
from models.lstm.ffs_dataset import FacesFeaturesDataModule

pl.seed_everything(42)


def prepare_data():
    x_train = pd.read_csv(CURRENT_FACES_FEATURES_DATA_X, delimiter=';')
    y_train = pd.read_csv(CURRENT_FACES_FEATURES_DATA_Y, delimiter=';')

    ffs_columns = x_train.columns.tolist()[1:]

    data = []
    for video_name, group in x_train.groupby('video_name'):
        faces_features = group[ffs_columns]
        auth = y_train[y_train.video_name == video_name].iloc[0].authenticity

        data.append((faces_features, auth))

    train_data, test_data = train_test_split(data, test_size=nns_conf.test_size)
    data_module = FacesFeaturesDataModule(train_data, test_data, batch_size=lstm_conf.batch_size)

    return data_module, ffs_columns
