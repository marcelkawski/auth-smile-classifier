import os
import sys
import numpy as np
import re
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_current_time_str


AUTH_SMILE_ENC_DICT = {
    'deliberate': 0,
    'spontaneous': 1
}

FACIAL_LANDMARKS_IDXS = {
    'jaw': (0, 17),
    'right_brow': (17, 22),
    'left_brow': (22, 27),
    'nose': (27, 36),
    'right_eye': (36, 42),
    'left_eye': (42, 48),
    'mouth': (48, 61),
    'lips': (61, 68)
}


def get_all_filenames(directory):
    return list(filter(None, [f for f in os.listdir(directory) if
                              os.path.isfile(os.path.join(directory, f)) and f[:2] != '._']))


def get_all_subdirs(directory):
    return list(filter(None, [f for f in os.listdir(directory)]))


def landmarks_to_np(shape, dtype='int'):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def get_frame_num(name):
    return int(re.search(r'frame(\d+)', name).group(1))


def get_filenames_sorted_by_frame_num(directory):
    return sorted(get_all_filenames(directory), key=lambda n: get_frame_num(n))


def save_dict_to_json_file(path, title, data):
    time_str = get_current_time_str()
    with open(os.path.abspath(os.path.join(os.sep, path, f'{title}-{time_str}.json')), 'w') as f:
        json.dump(data, f, indent=4)
        print('\nData successfully saved into the json file.\n')
