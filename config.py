import os

# -----PATHS-----
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data'))
VIDEOS_DATA_FILEPATH = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data',
                                                    'UvA-NEMO_Smile_Database_File_Details.csv'))
TEST_VIDEOS_DATA_FILEPATH = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data',
                                                         'UvA-NEMO_Smile_Database_File_Details_test2.csv'))
VIDEOS_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'videos'))
FRAMES_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'frames'))
FACES_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'faces'))
FACES_SAME_LEN_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'faces_same_len'))
NEW_FACES_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'new_faces'))
SMILES_DATA_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data_prep', 'smiles_data'))
SMILES_DATA_FILE_PATH = os.path.abspath(os.path.join(os.sep, SMILES_DATA_DIR, 'smiles_data-20221007-125320.json'))
COMPLETE_SMILES_DATA_FILE_PATH = os.path.abspath(os.path.join(os.sep, SMILES_DATA_DIR,
                                                              'complete_smiles_data-20221008-141009.json'))
NNS_WEIGHTS_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'models', 'nns', 'weights'))
NNS_PLOTS_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'models', 'nns', 'plots'))
NNS_LEARNING_DATA_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'models', 'nns', 'learning_data'))

# -----MODELS-----
# faces detection
FACES_DET_OPENCV_MODEL1_FP = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data_prep', 'models',
                                                          'haarcascade_frontalface_alt.xml'))
FACES_DET_OPENCV_MODEL2_FP = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data_prep', 'models',
                                                          'haarcascade_frontalface_alt2.xml'))

# faces features detection
FACES_FEATURES_DET_FP = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data_prep', 'models',
                                                     'shape_predictor_68_face_landmarks.dat'))

# -----EXTRACTING AND NORMALIZING FACES-----
DESIRED_FACE_WIDTH = 256
DESIRED_LEFT_EYE_POS = 0.35

FACES_FEATURES_WIDTH_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data',
                                                        f'faces_features{DESIRED_FACE_WIDTH}'))
FACES_FEATURES_DATA_WIDTH_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data',
                                                             f'faces_features_data{DESIRED_FACE_WIDTH}'))
FACES_FEATURES_DATA_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'faces_features_data'))
CUR_FFS_DATA_MODE = 'scaled'
CUR_FFS_DATA_NAME = 'lips_corners'
CURRENT_FACES_FEATURES_DATA_X = os.path.abspath(os.path.join(os.sep, ROOT_DIR, FACES_FEATURES_DATA_DIR,
                                                             f'{CUR_FFS_DATA_NAME}_{CUR_FFS_DATA_MODE}_x.csv'))
CURRENT_FACES_FEATURES_DATA_Y = os.path.abspath(os.path.join(os.sep, ROOT_DIR, FACES_FEATURES_DATA_DIR,
                                                             'authenticity.csv'))
CURRENT_FACES_FEATURES_DATA_TITLES = os.path.abspath(os.path.join(os.sep, ROOT_DIR, FACES_FEATURES_DATA_DIR,
                                                                  f'{CUR_FFS_DATA_NAME}_{CUR_FFS_DATA_MODE}_'
                                                                  f'titles.csv'))

LIPS_CORNER1_IDX = 48
LIPS_CORNER2_IDX = 54
FFS_DATA_CONFIG = {
    'features_name': 'lips_corners',
    'mode': 'scaled',  # 'scaled' / 'first'
    'features_nums': [LIPS_CORNER1_IDX, LIPS_CORNER2_IDX]
}

NUM_FACES_FEATURES = 68

BEG_SMILE_THRESHOLD = 1
END_SMILE_THRESHOLD = 0.05
NUM_FRAMES_RISE_SMILE_BEG = 20
MIN_DIFF_IN_RISE_SMILE_BEG = 0.01
SMILE_DURATION_MIN_RATIO = 0.48  # minimal <number_of_smile_frames>/<number_of_all_frames> ratio - If less than that
# - take from the beginning till the end
CURRENT_MIN_NUM_SMILE_FRAMES = 39  # number of frames of the shortest smile
