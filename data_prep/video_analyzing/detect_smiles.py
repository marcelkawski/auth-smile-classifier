import os
import sys
import cv2

import dlib
import numpy as np
import matplotlib.pyplot as plt
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FACES_FEATURES_DET_FP, FACES_DIR, LIPS_CORNER1_IDX, LIPS_CORNER2_IDX, BEG_SMILE_THRESHOLD, \
    END_SMILE_THRESHOLD, SMILES_DATA_DIR, SMILES_DATA_FILE_PATH, NUM_FRAMES_RISE_SMILE_BEG, \
    MIN_DIFF_IN_RISE_SMILE_BEG, SMILE_DURATION_MIN_RATIO
from data_prep.utils import get_all_subdirs, get_frame_num, get_filenames_sorted_by_frame_num


def show_smile_plot(data):
    frames = [d['frame'] for d in data]
    diffs = [d['diff'] for d in data]

    plt.figure(figsize=(18, 5))
    plt.plot(frames, diffs, '-o')

    plt.title('Zmiany odległości kącików ust od wartości początkowej w kolejnych klatkach')
    plt.xlabel('numer klatki')
    plt.ylabel('zmiana odległości kącików ust')

    plt.rcParams["figure.figsize"] = (200, 3)

    plt.show()


def save_smiles_data(show_plot=False, print_values=False, print_video_summary=False, sorted_by_num_smiles_frames=False):
    if sorted_by_num_smiles_frames is False:
        videos_names = get_all_subdirs(FACES_DIR)
    else:
        with open(SMILES_DATA_FILE_PATH, 'r') as fp:
            smiles_data = json.load(fp)
        frames = smiles_data['frames']
        videos_dicts = sorted(frames, key=lambda g: g['num_smiles_frames'])
        videos_names = [g['video_name'] for g in videos_dicts]

    if videos_names:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(FACES_FEATURES_DET_FP)

        smiles_frames = []

        for video_name in videos_names:
            faces_dir = os.path.abspath(os.path.join(os.sep, FACES_DIR, video_name))
            faces_names = get_filenames_sorted_by_frame_num(faces_dir)

            first_dist = None
            curr_diff = None
            diffs_in_time = []

            print(f'**********************************************\n{video_name}\n')

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

                    if print_values is True:
                        print(f'Frame num.: {_frame_number}\tdiff.: {curr_diff}')

                    diffs_in_time.append({
                        'frame': _frame_number,
                        'diff': curr_diff
                    })

            # finding top of the chart (to be able to find the end of the smile - It has to be after the top of
            # the chart.)
            filtered_diffs = [_dict for _dict in diffs_in_time if _dict['diff'] is not None]
            sorted_diffs = sorted(filtered_diffs, key=lambda d: d['diff'])
            biggest_diff_frame = sorted_diffs[-1]['frame']

            if print_values is True:
                print('\n')

            # finding beginning and end of the smile
            beg_found, end_found = False, False
            smile_beg_frame, smile_end_frame, num_smiles_frames = None, None, None

            try:
                for i in range(1, len(diffs_in_time)-1):  # From 1 because first value is always None.
                    dY = diffs_in_time[i+1]['diff'] - diffs_in_time[i]['diff']  # dX = 1 (frames difference)
                    if print_values is True:
                        print(f'Frame num.: {i+1}\tdY: {dY}')

                    diff = diffs_in_time[i+1]['diff']

                    rise_diffs = []
                    if beg_found is False:
                        for x in range(1, NUM_FRAMES_RISE_SMILE_BEG+1):
                            # IndexError can happen here:
                            rise_diffs.append(diffs_in_time[i+x]['diff'])

                        # print('\nframe: ', i)
                        # print('diff: ', diff)
                        # print('current diff ', diffs_in_time[i]['diff'])
                        # print('rise_diffs: ', rise_diffs)
                        # print(all(rise_diff > (diffs_in_time[i]['diff'] + MIN_DIFF_IN_RISE_SMILE_BEG) for rise_diff in
                        #       rise_diffs))
                        # print(dY > BEG_SMILE_THRESHOLD)
                        # print(dY)
                        # print(BEG_SMILE_THRESHOLD)

                    # print('\nframe: ', i)
                    # print('diff: ', diff)
                    # print('dY: ', dY)

                    if beg_found is False and dY > BEG_SMILE_THRESHOLD and \
                            all(rise_diff > (diffs_in_time[i]['diff'] + MIN_DIFF_IN_RISE_SMILE_BEG)
                                for rise_diff in rise_diffs):
                        # beginning of the smile found (slope of the line (differences between the current lips corners
                        # location and the lips corners location in the first frame) > BEG_SMILE_THRESHOLD
                        # - fast increase and then several values bigger than this point)
                        beg_found = True
                        smile_beg_frame = diffs_in_time[i]['frame']

                    elif beg_found is True and end_found is False and (END_SMILE_THRESHOLD * -1) < diff < \
                            END_SMILE_THRESHOLD and (i+1) > biggest_diff_frame:
                        # end of the smile found
                        # (between the current lips corners location and the lips corners location in the first frame
                        # close to 0 - return to the position at the beginning of the video and frame after the top of
                        # the chart)
                        end_found = True
                        smile_end_frame = diffs_in_time[i+1]['frame']
                        break

            except IndexError:
                print(f'\nError:\tNo smile beginning found in "{video_name}."\n')
                smile_beg_frame = 0

            if beg_found is False:
                print(f'\nError:\tNo smile beginning found in "{video_name}."\n')
                smile_beg_frame = 0
            if end_found is False:
                smile_end_frame = len(faces_names)  # last frame of the video

            num_smiles_frames = smile_end_frame - smile_beg_frame + 1
            num_frames = len(faces_names)

            if num_smiles_frames / num_frames < SMILE_DURATION_MIN_RATIO:  # SMILE_DURATION_MIN_RATIO - minimal
                # <number_of_smile_frames>/<number_of_all_frames> ratio - If less than that take from the beginning
                # till the end
                smile_end_frame = num_frames - 1
                num_smiles_frames = smile_end_frame - smile_beg_frame + 1

            smiles_frames.append({
                'video_name': video_name,
                'num_frames': num_frames,
                'smile_beg_frame': smile_beg_frame,
                'smile_end_frame': smile_end_frame,
                'num_smiles_frames': num_smiles_frames
            })

            if print_video_summary is True:
                print(f'\n'
                      f'number of frames: {num_frames}\n'
                      f'number of smile frames: {num_smiles_frames}\n\n'
                      f'smile beginning frame: {smile_beg_frame}\n'
                      f'smile end frame: {smile_end_frame}')

            if show_plot is True:
                show_smile_plot(diffs_in_time)

        smiles_data = {
            'smile_thresholds': {
                'beg': BEG_SMILE_THRESHOLD,
                'end': END_SMILE_THRESHOLD
            },
            'frames': smiles_frames
        }

        time_str = time.strftime("%Y%m%d-%H%M%S")

        with open(os.path.abspath(os.path.join(os.sep, SMILES_DATA_DIR, f'smiles_data-{time_str}.json')), 'w') as f:
            json.dump(smiles_data, f, indent=4)
            print('\nSmiles data successfully saved into the json file.\n')

    else:
        print('No faces to detect face features...')


if __name__ == '__main__':
    save_smiles_data(print_video_summary=True, sorted_by_num_smiles_frames=True, show_plot=True, print_values=True)
