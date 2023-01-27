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
NNS_WEIGHTS_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'models', 'video_nns', 'weights'))
NNS_PLOTS_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'models', 'video_nns', 'plots'))
NNS_LEARNING_DATA_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'models', 'video_nns', 'learning_data'))

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
DESIRED_FACE_PHOTO_WIDTH = 256
DESIRED_LEFT_EYE_POS = 0.35

NUM_FACES_FEATURES = 68
LIPS_CORNER1_IDX = 48
LIPS_CORNER2_IDX = 54
NOSE_TOP_IDX = 33
EYEBROWS_CORNERS_IDXS = [17, 21, 22, 26]


def create_face_features_nums():
    eyebrows = list(range(17, 27))
    nose = list(range(27, 36))
    eyes = list(range(36, 48))
    mouth = list(range(48, 68))

    all_points = sorted(eyebrows + eyes + nose + mouth)
    return all_points


FFS_DATA_ALT_MODES = ['lips_corners_from_nose_dist', 'lips_corners_from_nose_angle', 'lips_corners_dist']
FFS_DATA_CONFIG = {
    # 'features_name': 'lips_corners_dist',
    # 'features_name': 'lips_corners_from_nose_angle',
    # 'features_name': 'lips_corners_from_nose_dist',
    # 'features_name': 'lips_corners',
    # 'features_name': 'face',
    'features_name': 'eyebrows_corners',
    # 'features_name': 'all',
    # 'mode': 'scaled',  # 'scaled' / 'k_first_in_smile' / 'k_first'
    # 'mode': 'k_first_in_smile',
    'mode': 'k_first',
    # 'mode': 'all',
    # 'features_nums': [LIPS_CORNER1_IDX, LIPS_CORNER2_IDX]
    # 'features_nums': list(range(NUM_FACES_FEATURES))
    'features_nums': EYEBROWS_CORNERS_IDXS
    # 'features_nums': create_face_features_nums()
}
FACES_FEATURES_WIDTH_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data',
                                                        f'faces_features{DESIRED_FACE_PHOTO_WIDTH}'))
FACES_FEATURES_DATA_WIDTH_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data',
                                                             f'faces_features_data{DESIRED_FACE_PHOTO_WIDTH}'))
FACES_FEATURES_DATA_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'faces_features_data'))
CURRENT_FACES_FEATURES_DATA_X = os.path.abspath(os.path.join(os.sep, ROOT_DIR, FACES_FEATURES_DATA_DIR,
                                                             f'{FFS_DATA_CONFIG["features_name"]}_'
                                                             f'{FFS_DATA_CONFIG["mode"]}_x.csv'))
CURRENT_FACES_FEATURES_DATA_Y = os.path.abspath(os.path.join(os.sep, ROOT_DIR, FACES_FEATURES_DATA_DIR,
                                                             'authenticity.csv'))
CURRENT_FACES_FEATURES_DATA_TITLES = os.path.abspath(os.path.join(os.sep, ROOT_DIR, FACES_FEATURES_DATA_DIR,
                                                                  f'{FFS_DATA_CONFIG["features_name"]}_'
                                                                  f'{FFS_DATA_CONFIG["mode"]}_'
                                                                  f'titles.csv'))
FACES_FEATURES_DRAWINGS_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'faces_features_drawings'))
SMILES_DATA_PLOTS_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'smiles_data_plots'))
CURRENT_SMILES_DATA_PLOTS_DIR = os.path.abspath(os.path.join(os.sep, SMILES_DATA_PLOTS_DIR,
                                                             FFS_DATA_CONFIG['features_name']))

LIGHTNING_LOG_FILEPATH = os.path.abspath(os.path.join(os.sep, ROOT_DIR, FACES_FEATURES_DATA_DIR, 'lightning_logs'))
CLASSES = [0, 1]
CLASSES_STRS = ['0', '1']


def create_ffs_columns_names():
    ffs_cols_names = []
    if 'features_nums' in FFS_DATA_CONFIG:
        for feature_num in FFS_DATA_CONFIG['features_nums']:
            ffs_cols_names.append(f'{feature_num}x')
            ffs_cols_names.append(f'{feature_num}y')
    else:
        ffs_cols_names.append(FFS_DATA_CONFIG['features_name'])
    return ffs_cols_names


FFS_COLS_NAMES = create_ffs_columns_names()

BEG_SMILE_THRESHOLD = 1
END_SMILE_THRESHOLD = 0.05
NUM_FRAMES_RISE_SMILE_BEG = 20
MIN_DIFF_IN_RISE_SMILE_BEG = 0.01
SMILE_DURATION_MIN_RATIO = 0.48  # minimal <number_of_smile_frames>/<number_of_all_frames> ratio - If less than that
# - take from the beginning till the end
CURRENT_MIN_NUM_SMILE_FRAMES = 39  # number of frames of the shortest smile
SMILE_LABELS = ['zamierzony', 'spontaniczny']
SMILE_ORIGINAL_LABELS = ['deliberate', 'spontaneous']
