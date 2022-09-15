import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as img

from config import VIDEOS_DATA_FILEPATH, TEST_VIDEOS_DATA_FILEPATH, FRAMES_DIR
from models.nns.conf.video_analyzing import batch_size, test_size, video_frame_size, means, stds
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
    def __init__(self, data, frames_path=FRAMES_DIR, transform=None):
        super().__init__()
        self.data = data
        self.frames_path = frames_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_name, authenticity = self.data[index]
        frames_dir_path = os.path.abspath(os.path.join(os.sep, self.path, video_name))
        frames_names = get_filenames_sorted_by_frame_num(frames_dir_path)

        frames = []
        for frame_name in frames_names:
            frame_path = os.path.abspath(os.path.join(os.sep, frames_dir_path, frame_name))
            frame = img.imread(frame_path)
            if self.transform is not None:
                frame = self.transform(frame)
            frames.append(frame)

        return torch.stack(frames), torch.tensor(authenticity)

    def show_data_classes_sizes(self):
        print('Classes sizes: \n0- deliberate\n1- spontaneous\n')
        print(self.data['authenticity'].value_counts())

        labels = 'deliberate', 'spontaneous'
        plt.figure(figsize=(8, 8))
        plt.pie(self.data.groupby('authenticity').size(), labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Liczebność klas [%]')
        plt.show()


if __name__ == '__main__':
    _data = pd.read_csv(VIDEOS_DATA_FILEPATH, delimiter=';')
    vd = VideosDataset(data=_data)

    # vd.show_data_classes_sizes()

    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, stds)])

    test_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(means, stds)])

    valid_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, stds)])

    train_data, valid_data = train_test_split(_data, test_size=0.2)

