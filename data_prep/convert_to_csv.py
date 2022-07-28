import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep.utils import AUTH_SMILE_ENC_DICT
from config import DATA_DIR

if __name__ == "__main__":
    rows = open(os.path.abspath(os.path.join(os.sep, DATA_DIR, 'UvA-NEMO_Smile_Database_File_Details.txt')), 'r').\
        read().split('\n')
    data = [list(map(str.strip, row.split('\t'))) for row in rows[5:-1]]
    raw_data = {'video_filename': [row[0] for row in data],
                'authenticity': [AUTH_SMILE_ENC_DICT[row[4]] for row in data]}
    df = pd.DataFrame(raw_data, columns=['video_filename', 'authenticity'])
    df.to_csv(os.path.abspath(os.path.join(os.sep, DATA_DIR, 'UvA-NEMO_Smile_Database_File_Details.csv')), sep=';',
              index=False)
    print('Data successfully converted into the csv file.')
