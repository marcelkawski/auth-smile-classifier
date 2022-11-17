import os
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CURRENT_MIN_NUM_SMILE_FRAMES, CURRENT_FACES_FEATURES_DATA


class FacesFeaturesDataset(Dataset):
    def __init__(self, face_features_locs, authenticity, seq_len=CURRENT_MIN_NUM_SMILE_FRAMES):
        self.face_features_locs = face_features_locs.values
        self.authenticity = authenticity
        self.seq_len = seq_len

    def __len__(self):
        return self.face_features_locs.shape[0] // self.seq_len

    def __getitem__(self, index):
        return self.face_features_locs[index * self.seq_len: (index+1) * self.seq_len], self.authenticity[index]


if __name__ == '__main__':
    x = pd.read_csv(CURRENT_FACES_FEATURES_DATA, delimiter=';', header=None)

    train_dataset = FacesFeaturesDataset(x, [1] * 1235, seq_len=39)
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=False)

    for i, d in enumerate(train_loader):
        print(i, d[0], d[1])
        break
