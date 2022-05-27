import pandas as pd

from encode_dicts import AUTH_SMILE_ENC_DICT

if __name__ == "__main__":
    rows = open('./data/UvA-NEMO_Smile_Database_File_Details.txt', 'r').read().split('\n')
    data = [list(map(str.strip, row.split('\t'))) for row in rows[5:-1]]
    raw_data = {'video_filename': [row[0] for row in data],
                'authenticity': [AUTH_SMILE_ENC_DICT[row[4]] for row in data]}
    df = pd.DataFrame(raw_data, columns=['video_filename', 'authenticity'])
    df.to_csv('./data/UvA-NEMO_Smile_Database_File_Details.csv', sep=';', index=False)
    print('Data successfully converted into the csv file.')
