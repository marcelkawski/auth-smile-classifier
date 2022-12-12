import os
import sys
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CURRENT_FACES_FEATURES_DATA_X, CURRENT_FACES_FEATURES_DATA_Y, SMILE_LABELS

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ['#01BEFE', '#FFDD00', '#FF7D00', '#FF006D', '#ADFF02', '#8F00FF']
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 10, 8


def plot_auth_distr(col):
    vc = col.value_counts()
    plt.bar(SMILE_LABELS, vc.values)
    for i, val in enumerate(vc):
        plt.text(i, val+5, str(val), ha='center')
    plt.title('Rozkład autentyczności uśmiechów na filmach wideo')
    plt.show()


if __name__ == '__main__':
    x_train = pd.read_csv(CURRENT_FACES_FEATURES_DATA_X, delimiter=';')
    y_train = pd.read_csv(CURRENT_FACES_FEATURES_DATA_Y, delimiter=';')

    plot_auth_distr(y_train.authenticity)
