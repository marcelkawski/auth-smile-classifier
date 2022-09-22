import os
import sys
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FACES_FEATURES_DET_FP, FACES_DIR, FACES_SAME_LEN_DIR, LIPS_CORNER1_IDX, LIPS_CORNER2_IDX, \
    SMILE_THRESHOLD
from data_prep.utils import get_all_subdirs, get_frame_num, get_filenames_sorted_by_frame_num
from data_prep.frames_to_cut_dict import frames_to_cut_dict as ftcd


def show_plot(data):
    frames = [d['frame'] for d in data]
    diffs = [d['diff'] for d in data]

    plt.plot(frames, diffs, '-o')

    plt.title('Zmiany odległości kącików ust od wartości początkowej w kolejnych klatkach')
    plt.xlabel('numer klatki')
    plt.ylabel('zmiana odległości kącików ust')

    plt.show()


def get_frames_to_cut():
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

        frames_to_cut = []

        for video_name in todo_videos_names[:5]:
            # create dir for the faces of the same length
            faces_same_len_dir = os.path.abspath(os.path.join(os.sep, FACES_SAME_LEN_DIR, video_name))
            faces_dir = os.path.abspath(os.path.join(os.sep, FACES_DIR, video_name))
            faces_names = get_filenames_sorted_by_frame_num(faces_dir)
            
            first_dist = None
            curr_diff = None
            diffs_in_time = []

            if not os.path.exists(faces_same_len_dir):
                print(f'**********************************************\n{video_name}\n')

                # os.makedirs(faces_same_len_dir)

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

                        print(f'Frame num.: {_frame_number}\tdiff.: {curr_diff}')

                        diffs_in_time.append({
                            'frame': _frame_number,
                            'diff': curr_diff
                        })

                    # save an image with marked face features
                    # os.chdir(faces_same_len_dir)
                    # cv2.imwrite(face_name, img)

                beg_found, end_found = False, False
                smile_beg_frame, smile_end_frame, smile_duration = None, None, None

                for i in range(1, len(diffs_in_time)-1):  # From 1 because first value is always None.
                    dY = diffs_in_time[i+1]['diff'] - diffs_in_time[i]['diff']  # dX = 1 (frames difference)
                    diff = diffs_in_time[i+1]['diff']
                    
                    if beg_found is False and dY > SMILE_THRESHOLD:  # beginning of the smile found (slope of the
                        # line (differences between the current lips corners location and the lips corners location
                        # in the first frame) > SMILE_THRESHOLD - fast increase)
                        beg_found = True
                        smile_beg_frame = diffs_in_time[i]['frame']
                    
                    elif beg_found is True and end_found is False and -0.01 < diff < 0.01:  # end of the smile found
                        # (between the current lips corners location and the lips corners location in the first frame
                        # close to 0 - return to the position at the beginning of the video)
                        end_found = True
                        smile_end_frame = diffs_in_time[i+1]['frame']
                        break

                if beg_found is False:
                    print(f'\nError:\tNo smile beginning found in "{video_name}."\n')

                else:
                    if beg_found is True and end_found is False:
                        smile_end_frame = len(faces_names)  # last frame of the video

                    smile_duration = smile_end_frame - smile_beg_frame + 1

                frames_to_cut.append({
                    'video_name': video_name,
                    'smile_beg_frame': smile_beg_frame,
                    'smile_end_frame': smile_end_frame,
                    'smile_duration': smile_duration
                })

                print(f'smile beginning frame: {smile_beg_frame}\n'
                      f'smile end frame: {smile_end_frame}\n'
                      f'smile duration: {smile_duration}\n')

                show_plot(diffs_in_time)

        return frames_to_cut

    else:
        print('No faces to detect face features...')


if __name__ == '__main__':
    ftc = get_frames_to_cut()  # saved in file already
    # print(ftc)
    # print(ftc)
    # to_cut = min([f['num_frames_till_end'] for f in ftc])
    # print('We can cut: ', to_cut)

    # sorted_to_cut = sorted([f['num_frames_till_end'] for f in ftcd])
    # print(sorted_to_cut)
    #
    # to_cut = min([f['num_frames_till_end'] for f in ftcd])
    # print('We can cut: ', to_cut)

