import os
import sys
import pickle as pkl

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import NNS_LEARNING_DATA_DIR


def read_learning_data_file(file_name):
    file_path = os.path.abspath(os.path.join(os.sep, NNS_LEARNING_DATA_DIR, file_name))
    with open(file_path, 'rb') as inp:
        learning_data = pkl.load(inp)
    return learning_data


if __name__ == '__main__':
    ld = read_learning_data_file('CNN+LSTM_NN-20221108-212129.pkl')
    for d in ld:
        print(d)
