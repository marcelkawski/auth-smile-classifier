import os
import sys
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision.io import read_video

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as conf


class UvANemoSmileDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.videos_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.videos_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_name = os.path.join(self.root_dir, self.videos_data.iloc[idx, 0])
        video = read_video(video_name)
        authenticity = self.videos_data.iloc[idx, 1]
        sample = {'video': video, 'authenticity': authenticity}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    dataset = UvANemoSmileDataset(csv_file=conf.VIDEOS_DATA_FILEPATH, root_dir=conf.VIDEOS_DIR)
    smpl = dataset[5]
    print(smpl)
