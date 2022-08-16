import os
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_prep.utils import get_all_filenames, get_all_subdirs
from config import FACES_DIR


def get_frame_num(name):
    return int(re.search(r'frame(\d+)', name).group(1))


def test_consecutive_faces_frames_nums():
    faces_dirs = get_all_subdirs(FACES_DIR)
    for face_dirname in faces_dirs:
        face_dir = os.path.abspath(os.path.join(os.sep, FACES_DIR, face_dirname))
        # sorted by frame number
        faces_files = sorted(get_all_filenames(face_dir), key=lambda n: get_frame_num(n))
        for idx, name in enumerate(faces_files):
            frame_num = get_frame_num(name)
            assert frame_num == idx
