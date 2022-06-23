import os
import cv2

from config import VIDEOS_DIR, FRAMES_DIR
from data_prep.utils import get_all_videos_names

if __name__ == '__main__':
    videos_names = get_all_videos_names()
    frames_sets_created = 0

    for video_name in videos_names:
        print(f'**********************************************\n{video_name}\n')

        # create dir for the video's frames
        video_frames_dir = os.path.abspath(os.path.join(os.sep, FRAMES_DIR, video_name))
        if not os.path.exists(video_frames_dir):
            os.makedirs(video_frames_dir)

        current_frame = 0
        video_path = os.path.abspath(os.path.join(os.sep, VIDEOS_DIR, video_name))
        video = cv2.VideoCapture(video_path)

        # create frames
        os.chdir(video_frames_dir)
        frames_created = False
        while True:
            success, frame = video.read()
            if success:
                frames_created = True
                frame_name = f'{video_name}_frame{str(current_frame)}.jpg'
                print(f'Creating {frame_name}...')
                cv2.imwrite(frame_name, frame)
                current_frame += 1
            else:
                if frames_created:
                    frames_sets_created += 1
                break

        video.release()
        cv2.destroyAllWindows()

    print('Done!')
    print(f'Number of videos: {len(videos_names)}')
    print(f'Number of frames sets created: {frames_sets_created}')

