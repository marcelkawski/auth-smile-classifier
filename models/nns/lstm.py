import sys
import os
import torch
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models._config import nns_config as nns_conf
from models._config import LSTM_config as lstm_conf


class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.num_features = lstm_conf.num_features
        self.seq_len = lstm_conf.seq_length
        self.num_lstm_layers = lstm_conf.num_lstm_layers
        self.num_hidden = lstm_conf.num_hidden

        self.lstm = nn.LSTM(input_size=self.num_features,
                            hidden_size=self.num_hidden,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)
        self.hidden = None
        self.linear = nn.Linear(self.num_hidden * self.seq_len, nns_conf.num_classes)
        self.softmax = nn.Softmax(dim=0)

    def init_hidden(self, batch_size, device):
        hidden_state = torch.zeros(self.num_lstm_layers, batch_size, self.num_hidden, dtype=torch.float32).to(device)
        cell_state = torch.zeros(self.num_lstm_layers, batch_size, self.num_hidden, dtype=torch.float32).to(device)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.float()  # because of RuntimeError: "addmm_cuda" not implemented for 'Long'

        lstm_out, self.hidden = self.lstm(x, self.hidden)
        x = lstm_out.contiguous().view(batch_size, -1)
        y = self.linear(x)
        # y = self.softmax(y)
        return y
