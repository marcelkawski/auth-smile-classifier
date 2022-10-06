import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FACES_SAME_LEN_DIR, SMILES_DATA_FILE_PATH


if __name__ == '__main__':
    with open(SMILES_DATA_FILE_PATH, 'r') as fp:
        smiles_data = json.load(fp)
    min_num_smile_frames = sorted([f['num_smiles_frames'] for f in smiles_data['frames']])
    print(min_num_smile_frames)
    print(min_num_smile_frames[0])
