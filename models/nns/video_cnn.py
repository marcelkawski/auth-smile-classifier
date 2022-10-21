import sys
import os
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models._config import video_CNN1_config as vc1c
from models._config import nns_config as nc


class VideoCNN(nn.Module):
    def __init__(self):
        super(VideoCNN, self).__init__()
        self.conv1 = nn.Conv3d(39, 10, kernel_size=vc1c.kernel_size)
        self.conv2 = nn.Conv3d(10, 20, kernel_size=vc1c.kernel_size)
        self.conv2_drop = nn.Dropout3d()
        self.fc1 = nn.Linear(720, 1024)
        self.fc2 = nn.Linear(1024, nc.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool3d(self.conv1(x), 2))
        x = F.relu(F.max_pool3d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

