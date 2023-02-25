import sys
import os
import cv2
import dlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FRAMES_DIR, FACES_DIR, NEW_FACES_DIR, FACES_FEATURES_DET_FP, FACES_DET_OPENCV_MODEL1_FP, \
    FACES_DET_OPENCV_MODEL2_FP
from data_prep.data_prep_utils import get_all_subdirs, get_all_filenames
from data_prep.face_aligner import FaceAligner


def handle_arguments():
    arguments = sys.argv
    if len(arguments) != 2:
        raise Exception('Invalid number of parameters. This script accepts only ONE parameter - algorithm number.\n')

    arguments[1] = int(arguments[1])

    if arguments[1] not in [0, 1, 2]:
        raise Exception('Invalid extracting faces algorithm number.\n'
                        'Options to choose:\n'
                        '0: OpenCV: CascadeClassifier - haarcascade_frontalface_alt.xml (Viola-Jones algorithm)\n'
                        '1: OpenCV: CascadeClassifier - haarcascade_frontalface_alt2.xml (Viola-Jones algorithm)\n'
                        '2: Dlib: get_frontal_face_detector() (HOG + linear SVM algorithm)\n')

    return arguments


def get_faces(gray, algorithm):
    faces = None
    if algorithm == 0:
        face_cascade = cv2.CascadeClassifier(FACES_DET_OPENCV_MODEL1_FP)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    elif algorithm == 1:
        face_cascade = cv2.CascadeClassifier(FACES_DET_OPENCV_MODEL2_FP)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    elif algorithm == 2:
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)
    return faces


def save_faces(faces, face_name):
    faces_extracted = 0
    for num, face in enumerate(faces):
        if num != 0:
            cv2.imwrite(f'{face_name}_face{str(num)}.jpg', face)
        else:
            cv2.imwrite(face_name, face)
        faces_extracted += 1
    return faces_extracted


if __name__ == '__main__':
    _, alg_num = handle_arguments()

    if not os.path.exists(NEW_FACES_DIR):
        os.makedirs(NEW_FACES_DIR)

    videos_names = get_all_subdirs(FRAMES_DIR)
    done_videos_names = get_all_subdirs(FACES_DIR)
    # todo_videos_names = [vn for vn in videos_names if vn not in done_videos_names]
    todo_videos_names = videos_names

    print('all videos: ', len(videos_names))
    print('done videos: ', len(done_videos_names))
    print('videos to do: ', len(todo_videos_names))

    if todo_videos_names:
        predictor = dlib.shape_predictor(FACES_FEATURES_DET_FP)  # to detect features to align faces
        fa = FaceAligner(predictor)

        num_frames, dirs_created, _faces_extracted = 0, 0, 0

        for video_name in todo_videos_names:
            # create dir for the faces
            faces_dir = os.path.abspath(os.path.join(os.sep, NEW_FACES_DIR, video_name))
            frames_dir = os.path.abspath(os.path.join(os.sep, FRAMES_DIR, video_name))
            frames_names = get_all_filenames(frames_dir)
            video_faces_extracted = 0

            if not os.path.exists(faces_dir):
                print(f'**********************************************\n{video_name}\n')

                os.makedirs(faces_dir)
                dirs_created += 1
                num_frames += len(frames_names)

                os.chdir(faces_dir)

                for frame_name in frames_names:
                    frame_path = os.path.abspath(os.path.join(os.sep, frames_dir, frame_name))

                    img = cv2.imread(frame_path)
                    _gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _faces = get_faces(img, alg_num)

                    if _faces is not None:
                        # align and crop faces
                        aligned_faces = [fa.align(img, _gray, _face) for _face in _faces]
                        # save cropped faces
                        fe = save_faces(aligned_faces, frame_name)

                        video_faces_extracted += fe
                        _faces_extracted += fe

            print(f'Number of frames: {len(frames_names)}\nNumber of faces extracted: {video_faces_extracted}\nTo '
                  f'delete: {video_faces_extracted - len(frames_names)}')

        print('**********************************************\nDone!\n')
        # Numbers should be equal in pairs.
        print(f'Number of videos: {len(todo_videos_names)}')
        print(f'Number of created directories: {dirs_created}\n')

        print(f'Number of frames: {num_frames}')
        print(f'Number of extracted faces: {_faces_extracted}')
        print(f'Number of faces to delete: {_faces_extracted - num_frames}')

    else:
        print('Nothing to extract...')
