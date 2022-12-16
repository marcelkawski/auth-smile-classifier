import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_prep.data_prep_utils import get_all_filenames, get_all_subdirs
from config import FACES_DIR, FACES_FEATURES_WIDTH_DIR, FACES_FEATURES_DATA_WIDTH_DIR


def test_consecutive_frames_in_ffs_data_files():
    data_files_names = get_all_filenames(FACES_FEATURES_DATA_WIDTH_DIR)
    cols_list = ['frame_number']
    for dfn in data_files_names:
        path = os.path.abspath(os.path.join(os.sep, FACES_FEATURES_DATA_WIDTH_DIR, dfn))
        data = pd.read_csv(path, sep=';', usecols=cols_list)
        frame_numbers = list(data['frame_number'])
        print(dfn)
        for idx, frame_number in enumerate(frame_numbers):
            assert frame_number == idx


def test_created_data_names():
    ffs_dirs_names = sorted(get_all_subdirs(FACES_FEATURES_WIDTH_DIR))
    ffs_data_filenames = sorted([os.path.splitext(name)[0] for name in get_all_filenames(FACES_FEATURES_DATA_WIDTH_DIR)])
    assert ffs_data_filenames == ffs_dirs_names


def test_created_data_num():
    ffs_imgs_dirs_names = get_all_subdirs(FACES_FEATURES_WIDTH_DIR)
    for dirname in ffs_imgs_dirs_names:
        ffs_imgs_path = os.path.abspath(os.path.join(os.sep, FACES_FEATURES_WIDTH_DIR, dirname))
        ffs_data_file_path = os.path.abspath(os.path.join(os.sep, FACES_FEATURES_DATA_WIDTH_DIR, dirname + '.csv'))
        faces_path = os.path.abspath(os.path.join(os.sep, FACES_DIR, dirname))

        data = pd.read_csv(ffs_data_file_path, sep=';')

        created_imgs_num = len(get_all_filenames(ffs_imgs_path))
        created_data_rows = len(data)
        faces_num = len(get_all_filenames(faces_path))

        print(dirname, created_imgs_num, created_data_rows, faces_num)
        assert created_imgs_num == created_data_rows == faces_num
