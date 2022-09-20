import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as img

from config import VIDEOS_DATA_FILEPATH, TEST_VIDEOS_DATA_FILEPATH, FACES_DIR
from models.nns.conf.video_analyzing import batch_size, test_size, val_size, means, stds
from data_prep.utils import get_filenames_sorted_by_frame_num


def imshow(image, ax=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


class VideosDataset(Dataset):
    def __init__(self, data, frames_path=FACES_DIR, transform=None):
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
            frame = img.imread(frame_path)
            if self.transform is not None:
                frame = self.transform(frame)
            frames.append(frame)

        return frames, authenticity

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


if __name__ == '__main__':
    _data = pd.read_csv(TEST_VIDEOS_DATA_FILEPATH, delimiter=';')
    vd = VideosDataset(data=_data)

    # vd.show_data_classes_sizes()

    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, stds)])

    test_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(means, stds)])

    val_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means, stds)])

    _train_data, _test_data = train_test_split(_data, test_size=test_size)
    _train_data, _val_data = train_test_split(_train_data, test_size=val_size, random_state=1)

    _train_data = VideosDataset(_train_data, transform=train_transform)
    _val_data = VideosDataset(_val_data, transform=val_transform)
    _test_data = VideosDataset(_test_data, transform=test_transform)

    print_data_size(_data, _train_data, _val_data, _test_data)

    train_loader = DataLoader(dataset=_train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=_val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=_test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Model will be trained on: {device}\n')

    trainimages, trainlabels = next(iter(train_loader))

    print('1111')

    fig, axes = plt.subplots(figsize=(12, 12), ncols=5)
    print('training images')
    for i in range(5):
        print(i)
        axe1 = axes[i]
        imshow(trainimages[i], ax=axe1, normalize=False)

    plt.show()

    print(trainimages[0].size())
