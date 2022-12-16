import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config import FACES_SAME_LEN_DIR, COMPLETE_SMILES_DATA_FILE_PATH, CURRENT_MIN_NUM_SMILE_FRAMES
from data_prep.data_prep_utils import get_all_filenames, get_all_subdirs


def test_num_dirs():
    videos_names = get_all_subdirs(FACES_SAME_LEN_DIR)

    with open(COMPLETE_SMILES_DATA_FILE_PATH, 'r') as fp:
        smiles_data = json.load(fp)

    assert len(videos_names) == len(smiles_data['frames'])


def test_num_smile_frames():
    videos_names = get_all_subdirs(FACES_SAME_LEN_DIR)

    frames_lens = []
    for video_name in videos_names:
        frames_dir = os.path.abspath(os.path.join(os.sep, FACES_SAME_LEN_DIR, video_name))
        frames_lens.append(len(get_all_filenames(frames_dir)))
    frames_lens_vals = list(set(frames_lens))

    assert len(frames_lens_vals) == 1 and frames_lens_vals[0] == CURRENT_MIN_NUM_SMILE_FRAMES
