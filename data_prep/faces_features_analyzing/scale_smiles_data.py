import os
import sys
import json
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FACES_FEATURES_SCALED_DATA_DIR, FACES_FEATURES_DATA_DIR, COMPLETE_SMILES_DATA_FILE_PATH
from data_prep.utils import get_all_filenames


def scale_smiles_data(features_nums=None):
    if os.path.exists(FACES_FEATURES_SCALED_DATA_DIR) and os.listdir(FACES_FEATURES_SCALED_DATA_DIR):
        # Exists and is not empty.:
        raise Exception('Scaled faces features data directory is not empty so the program supposes that the faces '
                        'have been already extracted.\n')

    if not os.path.exists(FACES_FEATURES_SCALED_DATA_DIR):
        os.makedirs(FACES_FEATURES_SCALED_DATA_DIR)

    data_files_names = get_all_filenames(FACES_FEATURES_DATA_DIR)
    data_files_scaled = 0

    with open(COMPLETE_SMILES_DATA_FILE_PATH, 'r') as fp:
        smiles_data = json.load(fp)['frames']

    for data_file_name in data_files_names:
        print(data_file_name)
        scaled_data_filepath = os.path.abspath(os.path.join(os.sep, FACES_FEATURES_SCALED_DATA_DIR, data_file_name))
        data_filepath = os.path.abspath(os.path.join(os.sep, FACES_FEATURES_DATA_DIR, data_file_name))
        video_name = data_file_name.split('.csv')[0]
        scaled_frames_nums = [video['scaled_frames_nums'] for video in smiles_data if video['video_name'] ==
                              video_name][0]

        data = pd.read_csv(data_filepath, delimiter=';')
        selected_data = data[data['frame_number'].isin(scaled_frames_nums)]
        if features_nums is not None:
            selected_columns = [col_name for col_name in selected_data.columns if col_name != 'frame_number' and
                                int(col_name[:-1]) in features_nums]
            selected_data = selected_data[selected_columns]
        selected_data.to_csv(scaled_data_filepath, sep=';', index=False)
        data_files_scaled += 1

    print(f'Done! Successfully scaled {data_files_scaled} data files into the new csv files.')


if __name__ == '__main__':
    f_nums = [48, 54]
    scale_smiles_data(features_nums=f_nums)
