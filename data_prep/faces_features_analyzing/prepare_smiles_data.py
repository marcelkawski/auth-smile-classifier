import os
import sys
import json
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FACES_FEATURES_DATA_DIR, FACES_FEATURES_DATA_WIDTH_DIR, COMPLETE_SMILES_DATA_FILE_PATH, \
    FFS_DATA_CONFIG, FFS_DATA_ALT_MODES, NOSE_TOP_IDX, LIPS_CORNER1_IDX, LIPS_CORNER2_IDX, DESIRED_FACE_PHOTO_WIDTH
from data_prep.data_prep_utils import get_all_filenames


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def prepare_smiles_data(features_name, mode, features_nums=None):
    if not os.path.exists(FACES_FEATURES_DATA_DIR):
        os.makedirs(FACES_FEATURES_DATA_DIR)

    data_files_names = get_all_filenames(FACES_FEATURES_DATA_WIDTH_DIR)
    data_files_prepared = 0

    with open(COMPLETE_SMILES_DATA_FILE_PATH, 'r') as fp:
        smiles_data = json.load(fp)['frames']

    len_to_cut, min_num_frames = None, None
    if mode == 'k_first_in_smile':
        len_to_cut = min([(d['num_frames'] - d['smile_beg_frame'] - 1) for d in smiles_data])  # 39
    if mode == 'k_first':
        min_num_frames = min([d['num_frames'] for d in smiles_data])  # 55

    output_data_x_filepath = os.path.abspath(os.path.join(os.sep, FACES_FEATURES_DATA_DIR,
                                                          f'{features_name}_{mode}_x.csv'))

    for num, data_file_name in enumerate(data_files_names):
        print(data_file_name)
        data_x_filepath = os.path.abspath(os.path.join(os.sep, FACES_FEATURES_DATA_WIDTH_DIR, data_file_name))
        video_name = data_file_name.split('.csv')[0]

        data = pd.read_csv(data_x_filepath, delimiter=';')

        frames_nums = []
        if mode == 'scaled':
            frames_nums = [video['scaled_frames_nums'] for video in smiles_data if video['video_name'] ==
                           video_name][0]
        elif mode == 'k_first_in_smile':
            for video in smiles_data:
                if video['video_name'] == video_name:
                    smile_beg_frame = video['smile_beg_frame']
                    frames_nums = list(range(smile_beg_frame, smile_beg_frame + len_to_cut + 1))
                    break
        elif mode == 'k_first':
            frames_nums = list(range(min_num_frames))
        else:
            raise Exception('Incorrect "mode" given. Options to choose: "scaled" | "k_first_in_smile" | "k_first"\n')

        if frames_nums:
            selected_data_x = data[data['frame_number'].isin(frames_nums)]
        else:
            selected_data_x = data

        if features_nums is not None:
            selected_columns = [col_name for col_name in selected_data_x.columns if col_name != 'frame_number' and
                                int(col_name[:-1]) in features_nums]
            selected_data_x = selected_data_x[selected_columns]
        elif features_name in FFS_DATA_ALT_MODES:  # taking not only raw faces features points
            left_lips_corner_x = selected_data_x[f'{LIPS_CORNER1_IDX}x']
            left_lips_corner_y = DESIRED_FACE_PHOTO_WIDTH - selected_data_x[f'{LIPS_CORNER1_IDX}y']

            right_lips_corner_x = selected_data_x[f'{LIPS_CORNER2_IDX}x']
            right_lips_corner_y = DESIRED_FACE_PHOTO_WIDTH - selected_data_x[f'{LIPS_CORNER2_IDX}y']

            nose_top_x = selected_data_x[f'{NOSE_TOP_IDX}x']
            nose_top_y = DESIRED_FACE_PHOTO_WIDTH - selected_data_x[f'{NOSE_TOP_IDX}y']

            lc_dx = left_lips_corner_x - nose_top_x
            lc_dy = left_lips_corner_y - nose_top_y

            rc_dx = right_lips_corner_x - nose_top_x
            rc_dy = right_lips_corner_y - nose_top_y

            if features_name == 'lips_corners_from_nose_dist':
                lc_dist = np.sqrt((lc_dx ** 2) + (lc_dy ** 2))
                rc_dist = np.sqrt((rc_dx ** 2) + (rc_dy ** 2))
                av_dist = (lc_dist + rc_dist) / 2
                av_dist /= av_dist.iloc[0]
                selected_data_x[features_name] = av_dist
                selected_data_x = pd.DataFrame(selected_data_x[features_name])
            elif features_name == 'lips_corners_from_nose_angle':
                angles = []
                for i in range(len(lc_dx)):
                    v1 = lc_dx.iloc[i], lc_dy.iloc[i]
                    v2 = rc_dx.iloc[i], rc_dy.iloc[i]
                    angles.append(angle_between(v1, v2))
                angles = list(map(lambda el: el/angles[0], angles))
                selected_data_x = pd.DataFrame(angles, columns=[features_name])

        selected_data_x['video_name'] = video_name
        # set 'video_name' columns as first
        vn_col = selected_data_x['video_name']
        selected_data_x.drop(labels=['video_name'], axis=1, inplace=True)
        selected_data_x.insert(0, 'video_name', vn_col)

        if num != 0:
            selected_data_x.to_csv(output_data_x_filepath, mode='a', sep=';', index=False, header=False)
        else:
            selected_data_x.to_csv(output_data_x_filepath, sep=';', index=False, header=True)
        data_files_prepared += 1

    print(f'Done! Successfully scaled {data_files_prepared} data files into the new csv files.')


if __name__ == '__main__':
    # Change if needed.
    # prepare_smiles_data(FFS_DATA_CONFIG['features_name'], mode=FFS_DATA_CONFIG['mode'],
    #                     features_nums=FFS_DATA_CONFIG['features_nums'])
    prepare_smiles_data(FFS_DATA_CONFIG['features_name'], mode=FFS_DATA_CONFIG['mode'])
