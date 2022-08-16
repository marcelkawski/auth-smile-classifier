import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_prep.utils import get_all_filenames, get_all_subdirs
from config import FRAMES_DIR, FACES_DIR


def test_faces_num():
    videos_names = get_all_subdirs(FACES_DIR)
    for video_name in videos_names:
        num_frames = len(get_all_filenames(os.path.abspath(os.path.join(os.sep, FRAMES_DIR, video_name))))
        num_faces = len(get_all_filenames(os.path.abspath(os.path.join(os.sep, FACES_DIR, video_name))))
        print(video_name, num_frames, num_faces)
        assert num_faces == num_frames
