import os
import sys
import cv2
import dlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VIDEOS_DIR, FRAMES_DIR, FACES_DIR, FACES_DET_OPENCV_MODEL1_FP, FACES_DET_OPENCV_MODEL2_FP, \
    NEW_FACES_DIR
from data_prep.utils import get_all_filenames, get_all_subdirs


def handle_arguments():
    arguments = sys.argv
    if len(arguments) != 2:
        raise Exception('Invalid number of parameters. This script accepts only ONE parameter - algorithm number.')

    arguments[1] = int(arguments[1])

    if int(arguments[1]) not in [0, 1, 2]:
        raise Exception('Invalid extracting faces algorithm number.\n'
                        'Options to choose:\n'
                        '0: OpenCV haar-cascade haarcascade_frontalface_alt.xml\n'
                        '1: OpenCV haar-cascade haarcascade_frontalface_alt2.xml\n'
                        '2: dlib.get_frontal_face_detector\n')

    return arguments


def get_faces(image, algorithm):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if algorithm == 0:
        face_cascade = cv2.CascadeClassifier(FACES_DET_OPENCV_MODEL1_FP)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    elif algorithm == 1:
        face_cascade = cv2.CascadeClassifier(FACES_DET_OPENCV_MODEL2_FP)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    elif algorithm == 2:
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)
        return faces


def crop_faces(faces, image, face_name, algorithm):
    faces_extracted = 0
    if len(faces) == 1:
        if algorithm == 2:
            face = faces[0]
            cropped_face = image[face.top():face.bottom(), face.left():face.right()]
            cv2.imwrite(face_name, cropped_face)
        else:
            (x, y, w, h) = faces[0]
            cropped_face = image[y:y + h, x:x + w]
            cv2.imwrite(face_name, cropped_face)
        faces_extracted += 1
    else:
        for num, (x, y, w, h) in enumerate(faces):
            cropped_face = image[y:y + h, x:x + w]
            cv2.imwrite(f'{face_name}_face{str(num)}.jpg', cropped_face)
            faces_extracted += 1
    return faces_extracted


if __name__ == '__main__':
    _, alg_num = handle_arguments()

    if not os.path.exists(NEW_FACES_DIR):
        os.makedirs(NEW_FACES_DIR)

    videos_names = get_all_filenames(VIDEOS_DIR)
    done_videos_names = get_all_subdirs(FACES_DIR)
    todo_videos_names = [vn for vn in videos_names if vn not in done_videos_names]

    print('all videos: ', len(videos_names))
    print('done videos: ', len(done_videos_names))
    print('videos to do: ', len(todo_videos_names))
    
    if todo_videos_names:
        num_frames, dirs_created, _faces_extracted = 0, 0, 0

        for video_name in todo_videos_names:
            # create dir for the faces
            faces_dir = os.path.abspath(os.path.join(os.sep, NEW_FACES_DIR, video_name))
            frame_dir = os.path.abspath(os.path.join(os.sep, FRAMES_DIR, video_name))
            frames_names = get_all_filenames(frame_dir)
    
            if not os.path.exists(faces_dir):
                print(f'**********************************************\n{video_name}\n')

                os.makedirs(faces_dir)
                dirs_created += 1
                num_frames += len(frames_names)

                os.chdir(faces_dir)

                for frame_name in frames_names:
                    frame_path = os.path.abspath(os.path.join(os.sep, frame_dir, frame_name))

                    img = cv2.imread(frame_path)
                    _faces = get_faces(img, alg_num)
                    _faces_extracted += crop_faces(_faces, img, frame_name, alg_num)
    
            print(f'Number of frames: {len(frames_names)}\nNumber of faces extracted: {_faces_extracted}\nTo delete: '
                  f'{_faces_extracted - len(frames_names)}')
    
        print('**********************************************\nDone!\n')
        # Numbers should be equal in pairs.
        print(f'Number of videos: {len(todo_videos_names)}')
        print(f'Number of directories created: {dirs_created}\n')
        print(f'Number of frames: {num_frames}')
        print(f'Number of faces extracted: {_faces_extracted}')
        print(f'Number of faces to delete: {_faces_extracted - num_frames}')
        
    else:
        print('Nothing to extract...')
