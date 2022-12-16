import os
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FACES_DIR
from data_prep.data_prep_utils import get_all_subdirs, get_all_filenames


def rename_file(path, old_name, new_name):
    os.rename(os.path.abspath(os.path.join(os.sep, path, old_name)),
              os.path.abspath(os.path.join(os.sep, path, new_name)))


if __name__ == '__main__':
    if not os.path.exists(FACES_DIR) or not os.listdir(FACES_DIR):
        raise Exception('Faces directory does not exist or is empty so there is nothing to clean.\n')

    faces_dirs = get_all_subdirs(FACES_DIR)

    for face_dirname in faces_dirs:
        face_dir = os.path.abspath(os.path.join(os.sep, FACES_DIR, face_dirname))
        faces_files_to_clean = [filename for filename in get_all_filenames(face_dir) if 'face' in filename]
        for f in faces_files_to_clean:
            _, ext = os.path.splitext(f)
            rename_file(face_dir, f, re.sub(rf'_face\d{ext}', '', f))

    print('\nDone!')
