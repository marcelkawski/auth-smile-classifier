import os

# -----PATHS-----
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data'))
VIDEOS_DATA_FILEPATH = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data',
                                                    'UvA-NEMO_Smile_Database_File_Details.csv'))
TEST_VIDEOS_DATA_FILEPATH = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data',
                                                         'UvA-NEMO_Smile_Database_File_Details_test.csv'))
VIDEOS_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'videos'))
FRAMES_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'frames'))
FACES_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'faces'))
NEW_FACES_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'new_faces'))

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
DESIRED_FACE_WIDTH = 512
DESIRED_LEFT_EYE_POS = 0.35

FACES_FEATURES_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data',
                                                  f'faces_features{DESIRED_FACE_WIDTH}'))
FACES_FEATURES_DATA_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data',
                                                       f'faces_features_data{DESIRED_FACE_WIDTH}'))

# -----FACES FEATURES-----
NUM_FACES_FEATURES = 68
