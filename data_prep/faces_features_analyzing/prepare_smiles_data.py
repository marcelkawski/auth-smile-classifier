import os
import sys
import json
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FACES_FEATURES_DATA_DIR, FACES_FEATURES_DATA_WIDTH_DIR, COMPLETE_SMILES_DATA_FILE_PATH, \
    FFS_DATA_CONFIG
from data_prep.utils import get_all_filenames


def prepare_smiles_data(output_data_filename, mode, features_nums=None):
    # if os.path.exists(FACES_FEATURES_SCALED_DATA_DIR) and os.listdir(FACES_FEATURES_SCALED_DATA_DIR):
    #     # Exists and is not empty.:
    #     raise Exception('Scaled faces features data directory is not empty so the program supposes that the faces '
    #                     'have been already extracted.\n')

    if not os.path.exists(FACES_FEATURES_DATA_DIR):
        os.makedirs(FACES_FEATURES_DATA_DIR)

    data_files_names = get_all_filenames(FACES_FEATURES_DATA_WIDTH_DIR)
    data_files_prepared = 0

    with open(COMPLETE_SMILES_DATA_FILE_PATH, 'r') as fp:
        smiles_data = json.load(fp)['frames']

    len_to_cut = None
    if mode == 'k_first_in_smile':
        len_to_cut = min([(d['num_frames']-d['smile_beg_frame']-1) for d in smiles_data])
        print(len_to_cut)

    output_data_x_filepath = os.path.abspath(os.path.join(os.sep, FACES_FEATURES_DATA_DIR,
                                                          f'{output_data_filename}_{mode}_x.csv'))

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
                    frames_nums = list(range(smile_beg_frame, smile_beg_frame+len_to_cut+1))
                    break
        else:
            raise Exception('Incorrect "mode" given. Options to choose: "scaled" | "k_first_in_smile"\n')

        if frames_nums:
            selected_data_x = data[data['frame_number'].isin(frames_nums)]
        else:
            raise Exception('The algorithm cannot determine which frames numbers it should use.\n')

        if features_nums is not None:
            selected_columns = [col_name for col_name in selected_data_x.columns if col_name != 'frame_number' and
                                int(col_name[:-1]) in features_nums]
            selected_data_x = selected_data_x[selected_columns]
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
    prepare_smiles_data(FFS_DATA_CONFIG['features_name'], mode=FFS_DATA_CONFIG['mode'],
                        features_nums=FFS_DATA_CONFIG['features_nums'])
