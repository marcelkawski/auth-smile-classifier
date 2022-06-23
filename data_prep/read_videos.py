import os

from config import DATA_DIR


def get_all_folds_names():
    folds = []
    for i in range(1, 11):
        f = open(os.path.abspath(os.path.join(os.sep, DATA_DIR, 'experimental_protocols', 'fold_all',
                                              f'fold_all_{i}.txt')), 'r')
        lines = f.readlines()
        f.close()
        folds.append(list(map(str.strip, lines[2:])))
    print(folds)
    return folds
