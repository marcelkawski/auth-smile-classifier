from os import listdir
from os.path import isfile, join

from config import VIDEOS_DIR


def clean_name(name):
    if name[0:2] == '._':
        return name[2:]


def get_all_videos_names():
    return [clean_name(f) for f in listdir(VIDEOS_DIR) if isfile(join(VIDEOS_DIR, f))]


AUTH_SMILE_ENC_DICT = {
    'deliberate': 0,
    'spontaneous': 1
}
