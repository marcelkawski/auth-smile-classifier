import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FACES_SAME_LEN_DIR
from data_prep.utils import get_filenames_sorted_by_frame_num


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

