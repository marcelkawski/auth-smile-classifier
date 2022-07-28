import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VIDEOS_DIR, FRAMES_DIR, FACES_DIR, FACES_DET_OPENCV_FILEPATH
from data_prep.utils import get_all_filenames, get_dir_content


def get_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(FACES_DET_OPENCV_FILEPATH)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces


def crop_faces(faces, image, face_name):
    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        cropped_face = image[y:y + h, x:x + w]
        cv2.imwrite(face_name, cropped_face)
    else:
        for num, (x, y, w, h) in enumerate(faces):
            cropped_face = image[y:y + h, x:x + w]
            cv2.imwrite(f'{face_name}_face{str(num)}.jpg', cropped_face)


if __name__ == '__main__':
    if os.path.exists(FACES_DIR) and os.listdir(FACES_DIR):  # Exists and is not empty.
        print('Faces directory is not empty so the program supposes that the faces have been already extracted.\n')
        sys.exit()

    if not os.path.exists(FACES_DIR):
        os.makedirs(FACES_DIR)

    num_frames, dirs_created, faces_extracted = 0, 0, 0

    videos_names = get_all_filenames(VIDEOS_DIR)

    for video_name in videos_names:
        print(f'**********************************************\n{video_name}\n')

        # create dir for the faces
        video_faces_dir = os.path.abspath(os.path.join(os.sep, FACES_DIR, video_name))
        if not os.path.exists(video_faces_dir):
            os.makedirs(video_faces_dir)
            dirs_created += 1

        frame_dir = os.path.abspath(os.path.join(os.sep, FRAMES_DIR, video_name))
        frames_names = get_all_filenames(frame_dir)

        num_frames += len(frames_names)

        os.chdir(video_faces_dir)
        for frame_name in frames_names:
            print(f'Extracting from {frame_name}...')
            frame_path = os.path.abspath(os.path.join(os.sep, frame_dir, frame_name))
            img = cv2.imread(frame_path)
            _faces = get_faces(img)
            crop_faces(_faces, img, frame_name)
            faces_extracted += len(_faces)

    print('\nDone!\n')
    # Numbers should be equal in pairs.
    print(f'Number of videos: {len(videos_names)}')
    print(f'Number of directories created: {dirs_created}\n')
    print(f'Number of frames: {num_frames}')
    print(f'Number of faces extracted: {faces_extracted}')
    print(f'Number of faces to delete: {faces_extracted-num_frames}')
