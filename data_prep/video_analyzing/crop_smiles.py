import os
import sys
import json
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import COMPLETE_SMILES_DATA_FILE_PATH, FACES_SAME_LEN_DIR, FACES_DIR, CURRENT_MIN_NUM_SMILE_FRAMES


def create_complete_smiles_data_file():
    if os.path.exists(FACES_SAME_LEN_DIR) and os.listdir(FACES_SAME_LEN_DIR):  # Exists and is not empty.:
        raise Exception('Same length faces directory is not empty so the program supposes that the smiles have been '
                        'already cropped.\n')

    if not os.path.exists(FACES_SAME_LEN_DIR):
        os.makedirs(FACES_SAME_LEN_DIR)

    with open(COMPLETE_SMILES_DATA_FILE_PATH, 'r') as fp:
        smiles_data = json.load(fp)

    dirs_created = 0
    num_imgs_saved = []

    for video in smiles_data['frames']:
        video_name = video['video_name']
        print(f'**********************************************\n{video_name}\n')
        video_smile_dir = os.path.abspath(os.path.join(os.sep, FACES_SAME_LEN_DIR, video_name))
        video_faces_dir = os.path.abspath(os.path.join(os.sep, FACES_DIR, video_name))
        if not os.path.exists(video_smile_dir):
            os.makedirs(video_smile_dir)
            dirs_created += 1

        os.chdir(video_smile_dir)
        chosen_frames = video['scaled_frames_nums']
        n_imgs_saved = 0
        for frame_num in chosen_frames:
            img_title = f'{video_name}_frame{frame_num}.jpg'
            img_path = os.path.abspath(os.path.join(os.sep, video_faces_dir, img_title))
            img = cv2.imread(img_path)
            cv2.imwrite(img_title, img)
            n_imgs_saved += 1
        num_imgs_saved.append(n_imgs_saved)

    print(f'Done!\nNumber of created directories: {dirs_created}')
    num_imgs_saved_vals = list(set(num_imgs_saved))
    if len(num_imgs_saved_vals) == 1 and num_imgs_saved_vals[0] == CURRENT_MIN_NUM_SMILE_FRAMES:
        print(f'Success: All saved smiles frames have the same length ({num_imgs_saved_vals[0]}).')
    else:
        print(f'Error: Error during saving smiles frames. The do NOT have the same length. They should be '
              f'{CURRENT_MIN_NUM_SMILE_FRAMES} frames long.')


if __name__ == '__main__':
    create_complete_smiles_data_file()
