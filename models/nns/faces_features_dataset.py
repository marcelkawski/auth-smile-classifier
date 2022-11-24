import os
import sys
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CURRENT_MIN_NUM_SMILE_FRAMES


class FacesFeaturesDataset(Dataset):
    def __init__(self, face_features_locs, authenticity, titles, seq_len=CURRENT_MIN_NUM_SMILE_FRAMES):
        self.face_features_locs = face_features_locs.values
        self.authenticity = authenticity.values
        self.titles = titles.values
        self.seq_len = seq_len

    def __len__(self):
        return self.face_features_locs.shape[0] // self.seq_len

    def __getitem__(self, index):
        # print(self.titles[index])
        frames = torch.tensor(self.face_features_locs[index * self.seq_len: (index+1) * self.seq_len])
        authenticity = torch.tensor(self.authenticity[index])
        return frames, authenticity
