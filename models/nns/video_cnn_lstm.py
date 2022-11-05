import sys
import os
import torch.nn as nn
from torchvision import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models._config import nns_config as nns_conf
from models._config import CNNLSTM_config as cnn_lstm_conf


class VideoCNNLSTM(nn.Module):
    def __init__(self):
        super(VideoCNNLSTM, self).__init__()
        num_classes = nns_conf.num_classes
        dropout_p = cnn_lstm_conf.dropout_p
        lstm_hidden_size = cnn_lstm_conf.lstm_hidden_size
        lstm_num_layers = cnn_lstm_conf.lstm_num_layers

        baseModel = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()
        self.baseModel = baseModel
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(num_features, lstm_hidden_size, lstm_num_layers)
        self.fc1 = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch_size, num_frames, dim, h, w = x.shape

        i = 0
        y = self.baseModel((x[:, i]))
        output, (hn, cn) = self.lstm(y.unsqueeze(1))

        for i in range(1, num_frames):
            y = self.baseModel((x[:, i]))
            out, (hn, cn) = self.lstm(y.unsqueeze(1), (hn, cn))

        out = self.dropout(out[:, -1])
        out = self.fc1(out)

        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
