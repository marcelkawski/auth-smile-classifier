import os
import sys
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import VIDEOS_DATA_FILEPATH, CURRENT_FACES_FEATURES_DATA_X, CURRENT_FACES_FEATURES_DATA_Y, \
    CURRENT_FACES_FEATURES_DATA_TITLES, CURRENT_MIN_NUM_SMILE_FRAMES
from models.models_config import CNNLSTM_imgs_transforms_config as cnn_lstm_itc
from models.models_config import CNN3D_imgs_transforms_config as cnn_3d_itc
from models.models_config import nns_config as nns_conf
from models.video_nns.faces_features_dataset import FacesFeaturesDataset
from models.video_nns.videos_dataset import VideosDataset


def print_data_size(all_data, train_data, val_data, test_data, divide=False):
    if divide is False:
        all_data_len = len(all_data)
    else:
        all_data_len = int(len(all_data) / CURRENT_MIN_NUM_SMILE_FRAMES)
    train_data_len = len(train_data)
    val_data_len = len(val_data)
    test_data_len = len(test_data)

    print(f'\nData sizes:\n\n'
          f'all data: {all_data_len}\n'
          f'training data: {train_data_len} ({round(train_data_len / all_data_len * 100, 1)} %)\n'
          f'validation data: {val_data_len} ({round(val_data_len / all_data_len * 100, 1)} %)\n'
          f'test data: {test_data_len} ({round(test_data_len / all_data_len * 100, 1)} %)\n')


def prepare_datasets(num_model):
    _data, _train_data, _val_data, _test_data, transforms_dict = None, None, None, None, None
    divide = False

    if num_model == 0:
        transforms_dict = cnn_lstm_itc
    elif num_model == 1:
        transforms_dict = cnn_3d_itc

    _data = pd.read_csv(VIDEOS_DATA_FILEPATH, delimiter=';')

    train_transform = transforms.Compose([
        transforms.Resize((transforms_dict.h, transforms_dict.w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(transforms_dict.means, transforms_dict.stds),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((transforms_dict.h, transforms_dict.w)),
        transforms.ToTensor(),
        transforms.Normalize(transforms_dict.means, transforms_dict.stds),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((transforms_dict.h, transforms_dict.w)),
        transforms.ToTensor(),
        transforms.Normalize(transforms_dict.means, transforms_dict.stds),
    ])


    _train_data, _test_data = train_test_split(_data, test_size=nns_conf.test_size)
    _train_data, _val_data = train_test_split(_train_data, test_size=nns_conf.val_size)

    _train_data = VideosDataset(_train_data, transform=train_transform)
    _val_data = VideosDataset(_val_data, transform=val_transform)
    _test_data = VideosDataset(_test_data, transform=test_transform)

    print_data_size(_data, _train_data, _val_data, _test_data, divide=divide)
    return _train_data, _val_data, _test_data
