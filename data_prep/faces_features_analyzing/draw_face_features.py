import os
import sys
import cv2
import dlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FACES_FEATURES_DET_FP, FACES_DIR, NUM_FACES_FEATURES, FACES_FEATURES_DRAWINGS_DIR
from data_prep.data_prep_utils import get_frame_num, get_filenames_sorted_by_frame_num

f1 = lambda num: f'{num}x'
f2 = lambda num: f'{num}y'
header = ['frame_number'] + [f(i) for i in range(NUM_FACES_FEATURES) for f in (f1, f2)]


if __name__ == '__main__':
    videos_names = ['001_deliberate_smile_2.mp4']
    features_nums_to_circle = [48, 54, 33]

    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor(FACES_FEATURES_DET_FP)

    for video_name in videos_names:
        faces_features_dir = os.path.abspath(os.path.join(os.sep, FACES_FEATURES_DRAWINGS_DIR, video_name))
        faces_dir = os.path.abspath(os.path.join(os.sep, FACES_DIR, video_name))
        faces_names = get_filenames_sorted_by_frame_num(faces_dir)
        video_imgs_created = 0

        if not os.path.exists(faces_features_dir):
            print(f'**********************************************\n{video_name}\n')
            os.makedirs(faces_features_dir)

            for face_name in faces_names:
                face_path = os.path.abspath(os.path.join(os.sep, faces_dir, face_name))

                img = cv2.imread(face_path)
                gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
                _frame_number = get_frame_num(face_name)

                xs = []
                ys = []

                faces = detector(gray)
                for face in faces:
                    _landmarks = predictor(image=gray, box=face)

                    # draw on the image
                    for feature_num in features_nums_to_circle:
                        _x = _landmarks.part(feature_num).x
                        _y = _landmarks.part(feature_num).y
                        xs.append(_x)
                        ys.append(_y)
                        cv2.circle(img=img, center=(_x, _y), radius=5, color=(0, 255, 0), thickness=-1)

                    cv2.line(img=img, pt1=(xs[0], ys[0]), pt2=(xs[2], ys[2]), color=(0, 0, 255), thickness=2)
                    cv2.line(img=img, pt1=(xs[1], ys[1]), pt2=(xs[2], ys[2]), color=(0, 0, 255), thickness=2)

                # save an image with marked face features
                os.chdir(faces_features_dir)
                cv2.imwrite(face_name, img)
                video_imgs_created += 1

            print(f'Number of drawings created: {video_imgs_created}\n')

    print('**********************************************\nDone!\n')
