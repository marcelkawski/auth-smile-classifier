import os
import sys
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import CURRENT_FACES_FEATURES_DATA_X, CURRENT_FACES_FEATURES_DATA_Y

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ['#01BEFE', '#FFDD00', '#FF7D00', '#FF006D', '#ADFF02', '#8F00FF']
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 13, 8


def plot_auth_distr(col):
    col.value_counts().plot(kind='bar')
    plt.xticks(rotation=0)
    plt.title('Distribution of authenticity of smiles in videos')
    plt.show()


if __name__ == '__main__':
    x_train = pd.read_csv(CURRENT_FACES_FEATURES_DATA_X, delimiter=';')
    y_train = pd.read_csv(CURRENT_FACES_FEATURES_DATA_Y, delimiter=';')

    plot_auth_distr(y_train.authenticity)
