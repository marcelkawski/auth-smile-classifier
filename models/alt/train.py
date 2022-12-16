import os
os.environ["OMP_NUM_THREADS"] = '1'
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from pyts.multivariate.classification import MultivariateClassifier
from pyts.classification import KNeighborsClassifier, LearningShapelets, TimeSeriesForest, TSBF, BOSSVS, SAXVSM

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CURRENT_FACES_FEATURES_DATA_X, CURRENT_FACES_FEATURES_DATA_Y, FFS_COLS_NAMES
from models.models_config import nns_config as nns_conf


def handle_arguments():
    arguments = sys.argv
    if len(arguments) != 2:
        raise Exception('Invalid number of parameters. This script accepts only ONE parameter: multivariate time '
                        'series classifier.\n')

    arguments[1] = int(arguments[1])

    if arguments[1] not in [0, 1, 2, 3, 4, 5]:
        raise Exception('Invalid multivariate time series classifier.\n'
                        'Options to choose:\n'
                        '--------------------------------------------------\n'
                        '0: k-nearest neighbors classifier\n'
                        '1: Learning Shapelets algorithm\n'
                        '2: random forest classifier\n'
                        '3: Time Series Bag-of-Features algorithm\n'
                        '4: Bag-of-SFA Symbols in Vector Space\n'
                        '5: Classifier based on SAX-VSM representation and tf-idf statistics\n'
                        )

    return arguments


def create_data_lists(data):
    ffs_list, auths_list = [], []
    for x, y in data:
        ffs, auths = [], []

        for column in x:
            ffs.append(list(x[column].values))
        ffs_list.append(ffs)

        auths_list.append(y)

    return ffs_list, auths_list


def read_data():
    print(f'Learning with data from file: {CURRENT_FACES_FEATURES_DATA_X}\n')

    x = pd.read_csv(CURRENT_FACES_FEATURES_DATA_X, delimiter=';')
    y = pd.read_csv(CURRENT_FACES_FEATURES_DATA_Y, delimiter=';')

    data = []
    for video_name, group in x.groupby('video_name'):
        faces_features = group[FFS_COLS_NAMES]
        auth = y[y.video_name == video_name].iloc[0].authenticity
        data.append((faces_features, auth))

    return data


def split_data(data):
    train_data, test_data = train_test_split(data, test_size=nns_conf.test_size)

    x_train, y_train = create_data_lists(train_data)
    x_test, y_test = create_data_lists(test_data)

    return x_train, y_train, x_test, y_test


def train(x_train, y_train, algorithm_num):
    model = None
    if algorithm_num == 0:
        model = KNeighborsClassifier()
    elif algorithm_num == 1:
        model = LearningShapelets()
    elif algorithm_num == 2:
        model = TimeSeriesForest(max_features='sqrt', n_windows=0.5)
    elif algorithm_num == 3:
        model = TSBF(max_features='sqrt')
    elif algorithm_num == 4:
        model = BOSSVS()
    elif algorithm_num == 5:
        model = SAXVSM()

    clf = MultivariateClassifier(model)
    clf.fit(x_train, y_train)

    return clf


def test(clf, x_test, y_test):
    score = clf.score(x_test, y_test)
    return score


def get_av_accuracy(data, algorithm_num, k):
    accs = []
    for i in range(k):
        print(f'Learning for the {i+1}. time...')
        x_train, y_train, x_test, y_test = split_data(data)
        clf = train(x_train=x_train, y_train=y_train, algorithm_num=algorithm_num)
        acc = test(clf=clf, x_test=x_test, y_test=y_test)
        accs.append(acc)

    av_acc = sum(accs) / k
    print(f'Average accuracy: {av_acc}')


if __name__ == '__main__':
    _, alg_num = handle_arguments()
    d = read_data()
    get_av_accuracy(data=d, algorithm_num=alg_num, k=10)
