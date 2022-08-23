import os
import sys
import cv2
import dlib
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FACES_FEATURES_DET_FP, FACES_FEATURES_DIR, FACES_DIR, NUM_FACES_FEATURES, FACES_FEATURES_DATA_DIR
from data_prep.utils import get_all_subdirs, get_all_filenames, get_frame_num, get_filenames_sorted_by_frame_num

f1 = lambda num: f'{num}x'
f2 = lambda num: f'{num}y'
header = ['frame_number'] + [f(i) for i in range(NUM_FACES_FEATURES) for f in (f1, f2)]


def create_ff_data_file_writer(filepath):
    file = open(filepath, 'w', newline='')
    writer = csv.writer(file, delimiter=';')
    writer.writerow(header)
    return file, writer


def save_landmarks_row(writer, landmarks, frame_number):
    x = lambda n: landmarks.part(n).x
    y = lambda n: landmarks.part(n).y
    row = [frame_number] + [f(i) for i in range(NUM_FACES_FEATURES) for f in (x, y)]
    writer.writerow(row)


if __name__ == '__main__':
    if not os.path.exists(FACES_FEATURES_DIR):
        os.makedirs(FACES_FEATURES_DIR)

    if not os.path.exists(FACES_FEATURES_DATA_DIR):
        os.makedirs(FACES_FEATURES_DATA_DIR)

    videos_names = get_all_subdirs(FACES_DIR)
    done_videos_names = get_all_subdirs(FACES_FEATURES_DIR)
    todo_videos_names = [vn for vn in videos_names if vn not in done_videos_names][:2]

    if todo_videos_names:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(FACES_FEATURES_DET_FP)

        num_faces, dirs_created, data_files_created, imgs_created, rows_added = 0, 0, 0, 0, 0

        for video_name in todo_videos_names:
            # create dir for the faces features
            faces_features_dir = os.path.abspath(os.path.join(os.sep, FACES_FEATURES_DIR, video_name))
            faces_dir = os.path.abspath(os.path.join(os.sep, FACES_DIR, video_name))
            faces_names = get_filenames_sorted_by_frame_num(faces_dir)
            video_imgs_created = 0
            video_rows_added = 0

            if not os.path.exists(faces_features_dir):
                print(f'**********************************************\n{video_name}\n')

                os.makedirs(faces_features_dir)
                dirs_created += 1
                num_faces += len(faces_names)

                faces_features_data_filepath = os.path.abspath(os.path.join(os.sep, FACES_FEATURES_DATA_DIR,
                                                                            f'{video_name}.csv'))
                f, _writer = create_ff_data_file_writer(faces_features_data_filepath)

                for face_name in faces_names:
                    face_path = os.path.abspath(os.path.join(os.sep, faces_dir, face_name))

                    img = cv2.imread(face_path)
                    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
                    _frame_number = get_frame_num(face_name)

                    faces = detector(gray)
                    for face in faces:
                        # save in the data file
                        _landmarks = predictor(image=gray, box=face)

                        save_landmarks_row(_writer, _landmarks, _frame_number)
                        rows_added += 1
                        video_rows_added += 1
                        # draw on the image
                        for n in range(NUM_FACES_FEATURES):
                            _x = _landmarks.part(n).x
                            _y = _landmarks.part(n).y
                            cv2.circle(img=img, center=(_x, _y), radius=3, color=(0, 255, 0), thickness=-1)

                    # save an image with marked face features
                    os.chdir(faces_features_dir)
                    cv2.imwrite(face_name, img)
                    imgs_created += 1
                    video_imgs_created += 1

                f.close()
                print(f'{video_name}.csv face features data file created successfully.')
                data_files_created += 1
                print(f'Number of faces: {len(faces_names)}\nNumber of images created: {video_imgs_created}\n'
                      f'Number of rows added: {video_rows_added}')

        print('**********************************************\nDone!\n')
        print(f'Number of faces videos: {len(todo_videos_names)}')
        print(f'Number of created directories: {dirs_created}')
        print(f'Number of created data files: {data_files_created}\n')

        print(f'Number of faces: {num_faces}')
        print(f'Number of created images: {imgs_created}')
        print(f'Number of added rows: {rows_added}')

    else:
        print('No faces to detect face features...')
