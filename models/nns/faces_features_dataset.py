import os
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CURRENT_MIN_NUM_SMILE_FRAMES, CURRENT_FACES_FEATURES_DATA_X, CURRENT_FACES_FEATURES_DATA_Y, \
    CURRENT_FACES_FEATURES_DATA_TITLES
from models._config import nns_config


class FacesFeaturesDataset(Dataset):
    def __init__(self, face_features_locs, authenticity, titles, seq_len=CURRENT_MIN_NUM_SMILE_FRAMES):
        self.face_features_locs = face_features_locs.values
        self.authenticity = authenticity.values
        self.titles = titles.values
        self.seq_len = seq_len

    def __len__(self):
        return self.face_features_locs.shape[0] // self.seq_len

    def __getitem__(self, index):
        print(self.titles[index])
        return self.face_features_locs[index * self.seq_len: (index+1) * self.seq_len], self.authenticity[index]


def prepare_datasets(num_model):
    data_x = pd.read_csv(CURRENT_FACES_FEATURES_DATA_X, delimiter=';', header=None)
    data_y = pd.read_csv(CURRENT_FACES_FEATURES_DATA_Y, delimiter=';', header=None)
    data_titles = pd.read_csv(CURRENT_FACES_FEATURES_DATA_TITLES, delimiter=';', header=None)

    print(data_titles)

    # if num_model == 0:
    #     pass

    full_dataset = FacesFeaturesDataset(data_x, data_y, data_titles)
    dataset_size = len(full_dataset)

    val_ratio = round(nns_config.val_size * (1-nns_config.test_size), 3)
    val_size = round(val_ratio * dataset_size)
    train_size = round(round(1 - val_ratio - nns_config.test_size, 3) * dataset_size)
    test_size = round(nns_config.test_size * dataset_size)

    train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    for i, d in enumerate(train_loader):
        print(i, d[0], d[1])
        break

    return train_data, val_data, test_data


if __name__ == '__main__':
    prepare_datasets(0)
