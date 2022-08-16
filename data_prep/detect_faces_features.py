import sys
import os
import cv2
import dlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FACES_FEATURES_DET_FP, FRAMES_DIR
from data_prep.utils import get_all_subdirs, get_all_filenames

if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACES_FEATURES_DET_FP)

    # todo: checking if data file already exists and if not - create

    num_frames, rows_created = 0, 0
    frames_dirnames = get_all_subdirs(FRAMES_DIR)
    print('all frames: ', len(frames_dirnames))

    for frame_dirname in frames_dirnames:
        print(f'**********************************************\n{frame_dirname}\n')

        frame_dir = os.path.abspath(os.path.join(os.sep, FRAMES_DIR, frame_dirname))
        frames_names = get_all_filenames(frame_dir)

        num_frames += len(frames_names)

        for frame_name in frames_names:
            print(f'Detecting features from {frame_name}...')
            frame_path = os.path.abspath(os.path.join(os.sep, frame_dir, frame_name))

            img = cv2.imread(frame_path)
            gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(image=gray, box=face)
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    # todo: save row
                    cv2.circle(img=img, center=(x, y), radius=3, color=(0, 0, 255), thickness=-1)
