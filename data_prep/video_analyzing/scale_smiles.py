import os
import sys
import json
from fraction import Fraction

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import SMILES_DATA_FILE_PATH, SMILES_DATA_DIR
from data_prep.data_prep_utils import save_dict_to_json_file


def scale_frames(orig_len, req_len, beg_frame_num):
    fr = Fraction(orig_len, req_len)
    num, denom = fr.numerator, fr.denominator
    scaled_frames_nums = [i // denom + beg_frame_num for i in range(0, denom * orig_len, num)]
    return scaled_frames_nums


def create_complete_smiles_data_file():
    with open(SMILES_DATA_FILE_PATH, 'r') as fp:
        smiles_data = json.load(fp)
    min_num_smile_frames = min([f['num_smiles_frames'] for f in smiles_data['frames']])
    complete_frames_data, complete_smiles_data = [], {}
    for idx, sd in enumerate(smiles_data['frames']):
        _scaled_frames_nums = scale_frames(sd['num_smiles_frames'], min_num_smile_frames, sd['smile_beg_frame'])
        sd['scaled_frames_nums'] = _scaled_frames_nums
        complete_frames_data.append(sd)

    complete_smiles_data = {
        'smiles_data_file': rf'{SMILES_DATA_FILE_PATH}',
        'frames': complete_frames_data
    }

    scaled_frames_lens = [len(d['scaled_frames_nums']) for d in complete_smiles_data['frames']]
    lens_values = list(set(scaled_frames_lens))
    if len(lens_values) == 1 and lens_values[0] == min_num_smile_frames:
        print(f'Success: All videos have the same number of scaled frames numbers ({min_num_smile_frames}).')
        save_dict_to_json_file(SMILES_DATA_DIR, 'complete_smiles_data', complete_smiles_data)
    else:
        print(f'Error: All videos does NOT have the same number of scaled frames numbers. They should have: '
              f'{min_num_smile_frames}.')


if __name__ == '__main__':
    create_complete_smiles_data_file()
