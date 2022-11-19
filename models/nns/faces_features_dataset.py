import os
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CURRENT_MIN_NUM_SMILE_FRAMES, CURRENT_FACES_FEATURES_DATA_X
from models._config import nns_config


class FacesFeaturesDataset(Dataset):
    def __init__(self, face_features_locs, authenticity, seq_len=CURRENT_MIN_NUM_SMILE_FRAMES):
        self.face_features_locs = face_features_locs.values
        self.authenticity = authenticity
        self.seq_len = seq_len

    def __len__(self):
        return self.face_features_locs.shape[0] // self.seq_len

    def __getitem__(self, index):
        return self.face_features_locs[index * self.seq_len: (index+1) * self.seq_len], self.authenticity[index]


def prepare_datasets(num_model):
    data = pd.read_csv(CURRENT_FACES_FEATURES_DATA_X, delimiter=';', header=None)
    print(type(data))
    print(data)

    # if num_model == 0:
    #     pass
    #
    # full_dataset = FacesFeaturesDataset(data, [1] * 1235, seq_len=39)
    # dataset_size = len(full_dataset)
    #
    # val_ratio = round(nns_config.val_size * (1-nns_config.test_size), 3)
    # val_size = round(val_ratio * dataset_size)
    # train_size = round(round(1 - val_ratio - nns_config.test_size, 3) * dataset_size)
    # test_size = round(nns_config.test_size * dataset_size)
    #
    # train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    #
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    #
    # for i, d in enumerate(train_loader):
    #     print(d[0].shape, d[1].shape)
    #     print(d[0], d[1])
    #     break

    # train_data = FacesFeaturesDataset(train_data)
    # val_data = FacesFeaturesDataset(val_data)
    # test_data = FacesFeaturesDataset(test_data)
    #
    # print_data_size(data, train_data, val_data, test_data)
    #
    # return train_data, val_data, test_data


if __name__ == '__main__':
    prepare_datasets(0)
    # x = pd.read_csv(CURRENT_FACES_FEATURES_DATA_X, delimiter=';', header=None)
    # train_dataset = FacesFeaturesDataset(x, [1] * 1235, seq_len=39)
    # train_loader = DataLoader(train_dataset, batch_size=3, shuffle=False)
    #
    # for i, d in enumerate(train_loader):
    #     print(i, d[0], d[1])
    #     break
