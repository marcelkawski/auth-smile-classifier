import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CURRENT_FACES_FEATURES_DATA_X, CURRENT_MIN_NUM_SMILE_FRAMES


def test_frames_num_after_scaling():
    x_train = pd.read_csv(CURRENT_FACES_FEATURES_DATA_X, delimiter=';')
    val_counts = x_train['video_name'].value_counts()
    len_to_compare = len(val_counts)
    assert len(val_counts[val_counts == CURRENT_MIN_NUM_SMILE_FRAMES]) == len_to_compare
