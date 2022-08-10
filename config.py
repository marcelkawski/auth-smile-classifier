import os

# paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data'))
VIDEOS_DATA_FILEPATH = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data',
                                                    'UvA-NEMO_Smile_Database_File_Details.csv'))
VIDEOS_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'videos'))
FRAMES_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'frames'))
FACES_DIR = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data', 'faces'))
FACES_DET_OPENCV_FILEPATH = os.path.abspath(os.path.join(os.sep, ROOT_DIR, 'data_prep', 'models_xmls',
                                                         'haarcascade_frontalface_alt.xml'))

