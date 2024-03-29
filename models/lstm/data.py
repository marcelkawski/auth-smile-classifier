import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CURRENT_FACES_FEATURES_DATA_X, CURRENT_FACES_FEATURES_DATA_Y, FFS_COLS_NAMES
from models.models_config import nns_config as nns_conf
from models.models_config import LSTM_config as lstm_conf
from models.lstm.dataset import FacesFeaturesDataModule



def prepare_data():
    x_train = pd.read_csv(CURRENT_FACES_FEATURES_DATA_X, delimiter=';')
    y_train = pd.read_csv(CURRENT_FACES_FEATURES_DATA_Y, delimiter=';')

    print(f'Learning with data from file: {CURRENT_FACES_FEATURES_DATA_X}\n')

    data = []
    for video_name, group in x_train.groupby('video_name'):
        faces_features = group[FFS_COLS_NAMES]
        auth = y_train[y_train.video_name == video_name].iloc[0].authenticity

        data.append((faces_features, auth))

    train_data, test_data = train_test_split(data, test_size=nns_conf.test_size)
    data_module = FacesFeaturesDataModule(train_data, test_data, batch_size=lstm_conf.batch_size)

    return data_module, test_data


if __name__ == '__main__':
    prepare_data()
