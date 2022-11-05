import os
import sys
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import VIDEOS_DATA_FILEPATH, FACES_SAME_LEN_DIR
from data_prep.utils import get_filenames_sorted_by_frame_num
from models._config import CNNLSTM_imgs_transforms_config as cnn_lstm_itc
from models._config import nns_config as nc


class VideosDataset(Dataset):
    def __init__(self, data, frames_path=FACES_SAME_LEN_DIR, transform=None):
        super().__init__()
        self.data = data.values
        self.frames_path = frames_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_name, authenticity = self.data[index]
        frames_dir_path = os.path.abspath(os.path.join(os.sep, self.frames_path, video_name))
        frames_names = get_filenames_sorted_by_frame_num(frames_dir_path)

        frames = []
        for frame_name in frames_names:
            frame_path = os.path.abspath(os.path.join(os.sep, frames_dir_path, frame_name))
            frame = Image.open(frame_path)
            if self.transform is not None:
                frame = self.transform(frame)
            frames.append(frame)

        return torch.stack(frames), authenticity

    def show_data_classes_sizes(self):
        print('Classes sizes: \n0- deliberate\n1- spontaneous\n')
        print(self.data['authenticity'].value_counts())

        labels = 'deliberate', 'spontaneous'
        plt.figure(figsize=(8, 8))
        plt.pie(self.data.groupby('authenticity').size(), labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Liczebność klas [%]')
        plt.show()


def print_data_size(all_data, train_data, val_data, test_data):
    all_data_len = len(all_data)
    train_data_len = len(train_data)
    val_data_len = len(val_data)
    test_data_len = len(test_data)

    print(f'\nLengths:\n\n'
          f'all data: {all_data_len}\n'
          f'training data: {train_data_len} ({round(train_data_len / all_data_len * 100, 1)} %)\n'
          f'validation data: {val_data_len} ({round(val_data_len / all_data_len * 100, 1)} %)\n'
          f'test data: {test_data_len} ({round(test_data_len / all_data_len * 100, 1)} %)\n')


def denormalize(x_, means, stds):
    x = x_.clone()
    for i in range(3):
        x[i] = x[i]*stds[i]+means[i]
    x = to_pil_image(x)
    return x


def prepare_datasets():
    _data = pd.read_csv(VIDEOS_DATA_FILEPATH, delimiter=';')

    # vd = VideosDataset(data=_data)
    # vd.show_data_classes_sizes()

    train_transform = transforms.Compose([
        transforms.Resize((cnn_lstm_itc.h, cnn_lstm_itc.w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(cnn_lstm_itc.means, cnn_lstm_itc.stds),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((cnn_lstm_itc.h, cnn_lstm_itc.w)),
        transforms.ToTensor(),
        transforms.Normalize(cnn_lstm_itc.means, cnn_lstm_itc.stds),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((cnn_lstm_itc.h, cnn_lstm_itc.w)),
        transforms.ToTensor(),
        transforms.Normalize(cnn_lstm_itc.means, cnn_lstm_itc.stds),
    ])

    _train_data, _test_data = train_test_split(_data, test_size=nc.test_size)
    _train_data, _val_data = train_test_split(_train_data, test_size=nc.val_size, random_state=1)

    _train_data = VideosDataset(_train_data, transform=train_transform)
    _val_data = VideosDataset(_val_data, transform=val_transform)
    _test_data = VideosDataset(_test_data, transform=test_transform)

    print_data_size(_data, _train_data, _val_data, _test_data)

    return _train_data, _val_data, _test_data
