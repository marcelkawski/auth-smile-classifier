import os
import sys
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FACES_FEATURES_DET_FP, FACES_DIR, FACES_SAME_LEN_DIR, LIPS_CORNER1_IDX, LIPS_CORNER2_IDX, \
    SMILE_BEG_THRESHOLD
from data_prep.utils import get_all_subdirs, get_frame_num, get_filenames_sorted_by_frame_num


def show_plot(data):
    frames = [d['frame'] for d in data]
    diffs = [d['diff'] for d in data]

    plt.plot(frames, diffs, '-o')

    plt.title('Zmiany odległości kącików ust od wartości początkowej w kolejnych klatkach')
    plt.xlabel('numer klatki')
    plt.ylabel('zmiana odległości kącików ust')

    plt.show()


if __name__ == '__main__':
    if not os.path.exists(FACES_SAME_LEN_DIR):
        os.makedirs(FACES_SAME_LEN_DIR)

    videos_names = get_all_subdirs(FACES_DIR)
    done_videos_names = get_all_subdirs(FACES_SAME_LEN_DIR)
    todo_videos_names = [vn for vn in videos_names if vn not in done_videos_names]

    print('all videos: ', len(videos_names))
    print('done videos: ', len(done_videos_names))
    print('videos to do: ', len(todo_videos_names))

    if todo_videos_names:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(FACES_FEATURES_DET_FP)

        for video_name in todo_videos_names[:10]:
            # create dir for the faces of the same length
            faces_same_len_dir = os.path.abspath(os.path.join(os.sep, FACES_SAME_LEN_DIR, video_name))
            faces_dir = os.path.abspath(os.path.join(os.sep, FACES_DIR, video_name))
            faces_names = get_filenames_sorted_by_frame_num(faces_dir)
            
            first_dist = None
            curr_diff = None
            diffs_in_time = []

            if not os.path.exists(faces_same_len_dir):
                print(f'**********************************************\n{video_name}\n')

                os.makedirs(faces_same_len_dir)

                for face_name in faces_names:
                    face_path = os.path.abspath(os.path.join(os.sep, faces_dir, face_name))

                    img = cv2.imread(face_path)
                    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
                    _frame_number = get_frame_num(face_name)

                    faces = detector(gray)
                    for face in faces:
                        _landmarks = predictor(image=gray, box=face)

                        x1 = _landmarks.part(LIPS_CORNER1_IDX).x
                        y1 = _landmarks.part(LIPS_CORNER1_IDX).y

                        x2 = _landmarks.part(LIPS_CORNER2_IDX).x
                        y2 = _landmarks.part(LIPS_CORNER2_IDX).y

                        dY = y2 - y1
                        dX = x2 - x1
                        dist = np.sqrt((dX ** 2) + (dY ** 2))

                        if first_dist is None:
                            first_dist = dist
                        else:
                            curr_diff = abs(dist - first_dist)

                        cv2.circle(img=img, center=(x1, y1), radius=3, color=(0, 255, 0), thickness=-1)
                        cv2.circle(img=img, center=(x2, y2), radius=3, color=(0, 255, 0), thickness=-1)

                        # print(f'Frame num.: {_frame_number}\tdiff.: {curr_diff}')

                        diffs_in_time.append({
                            'frame': _frame_number,
                            'diff': curr_diff
                        })

                    # save an image with marked face features
                    os.chdir(faces_same_len_dir)
                    cv2.imwrite(face_name, img)

                # show_plot(diffs_in_time)

                coefs_in_time = []
                for i in range(1, len(diffs_in_time)-1):  # From 1 because first value is always None.
                    dY = diffs_in_time[i+1]['diff'] - diffs_in_time[i]['diff']  # dX = 1 (frames difference)
                    if dY > SMILE_BEG_THRESHOLD:
                        smile_beg_frame = diffs_in_time[i]['frame']
                        break

        print('\nDone!\n')

    else:
        print('No faces to detect face features...')
