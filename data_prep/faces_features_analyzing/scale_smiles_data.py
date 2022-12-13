import os
import sys
import json
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FACES_FEATURES_DATA_DIR, FACES_FEATURES_DATA_WIDTH_DIR, COMPLETE_SMILES_DATA_FILE_PATH, \
    VIDEOS_DATA_FILEPATH
from data_prep.utils import get_all_filenames


def scale_smiles_data(scaled_data_filename, features_nums=None):
    if os.path.exists(FACES_FEATURES_DATA_DIR) and os.listdir(FACES_FEATURES_DATA_DIR):
        # Exists and is not empty.:
        raise Exception('Scaled faces features data directory is not empty so the program supposes that the faces '
                        'have been already extracted.\n')

    if not os.path.exists(FACES_FEATURES_DATA_DIR):
        os.makedirs(FACES_FEATURES_DATA_DIR)

    data_files_names = get_all_filenames(FACES_FEATURES_DATA_WIDTH_DIR)
    data_files_scaled = 0

    with open(COMPLETE_SMILES_DATA_FILE_PATH, 'r') as fp:
        smiles_data = json.load(fp)['frames']
    videos_data = pd.read_csv(VIDEOS_DATA_FILEPATH, delimiter=';')
    authenticities = []

    scaled_data_x_filepath = os.path.abspath(os.path.join(os.sep, FACES_FEATURES_DATA_DIR,
                                                          f'{scaled_data_filename}_x.csv'))
    scaled_data_y_filepath = os.path.abspath(os.path.join(os.sep, FACES_FEATURES_DATA_DIR,
                                                          f'{scaled_data_filename}_y.csv'))
    scaled_data_titles_filepath = os.path.abspath(os.path.join(os.sep, FACES_FEATURES_DATA_DIR,
                                                               f'{scaled_data_filename}_titles.csv'))

    for data_file_name in data_files_names:
        print(data_file_name)
        data_x_filepath = os.path.abspath(os.path.join(os.sep, FACES_FEATURES_DATA_WIDTH_DIR, data_file_name))
        video_name = data_file_name.split('.csv')[0]

        authenticity = videos_data.loc[videos_data['video_filename'] == video_name, 'authenticity'].iloc[0]
        authenticities.append(authenticity)

        scaled_frames_nums = [video['scaled_frames_nums'] for video in smiles_data if video['video_name'] ==
                              video_name][0]

        data = pd.read_csv(data_x_filepath, delimiter=';')
        selected_data_x = data[data['frame_number'].isin(scaled_frames_nums)]
        if features_nums is not None:
            selected_columns = [col_name for col_name in selected_data_x.columns if col_name != 'frame_number' and
                                int(col_name[:-1]) in features_nums]
            selected_data_x = selected_data_x[selected_columns]
        selected_data_x.to_csv(scaled_data_x_filepath, mode='a', sep=';', index=False, header=False)
        data_files_scaled += 1

    authenticities = pd.DataFrame(authenticities)
    authenticities.to_csv(scaled_data_y_filepath, sep=';', index=False, header=False)
    titles = pd.DataFrame(data_files_names)
    titles.to_csv(scaled_data_titles_filepath, sep=';', index=False, header=False)

    print(f'Done! Successfully scaled {data_files_scaled} data files into the new csv files.')


if __name__ == '__main__':
    # Change if needed.
    sc_data_filename = 'lips_corners'
    f_nums = [48, 54]

    scale_smiles_data(sc_data_filename, features_nums=f_nums)
