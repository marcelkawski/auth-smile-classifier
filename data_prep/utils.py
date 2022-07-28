import os
import sys
from os import listdir
from os.path import isfile, join

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_all_filenames(directory):
    return list(filter(None, [f for f in listdir(directory) if isfile(join(directory, f)) and f[:2] != '._']))


def get_dir_content(directory):
    return list(filter(None, [f for f in listdir(directory)]))


AUTH_SMILE_ENC_DICT = {
    'deliberate': 0,
    'spontaneous': 1
}
