import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FACES_FEATURES_DATA_WIDTH_DIR, CURRENT_FACES_FEATURES_DATA_Y
from data_prep.data_prep_utils import get_all_subdirs


def check_incorr_detections():
    dirs_video_names = [name.split('.csv')[0] for name in get_all_subdirs(FACES_FEATURES_DATA_WIDTH_DIR)]
    auths_video_names = pd.read_csv(CURRENT_FACES_FEATURES_DATA_Y, delimiter=';')['video_name'].values
    diffs = list(set(auths_video_names).difference(dirs_video_names))
    print(diffs)


if __name__ == '__main__':
    check_incorr_detections()
