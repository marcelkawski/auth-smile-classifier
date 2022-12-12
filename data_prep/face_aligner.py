import sys
import os
import numpy as np
import cv2
import dlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep.utils import landmarks_to_np, FACIAL_LANDMARKS_IDXS
from config import DESIRED_FACE_PHOTO_WIDTH
from config import DESIRED_LEFT_EYE_POS as DLAP


class FaceAligner:
    def __init__(self, predictor, desired_left_eye_pos=(DLAP, DLAP), desired_face_photo_width=DESIRED_FACE_PHOTO_WIDTH,
                 desired_face_photo_height=None):
        self.predictor = predictor
        self.desired_left_eye_pos = desired_left_eye_pos
        self.desired_face_photo_width = desired_face_photo_width
        self.desired_face_photo_height = desired_face_photo_height
        if self.desired_face_photo_height is None:
            self.desired_face_photo_height = self.desired_face_photo_width

    def align(self, image, gray, rect):
        if isinstance(rect, np.ndarray) and len(rect) == 4:  # data format for first 2 algorithms extracting faces
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]
            rect = dlib.rectangle(x, y, x + w, y + h)
        shape = self.predictor(gray, rect)
        shape = landmarks_to_np(shape)
        (left_eye_beg, left_eye_end) = FACIAL_LANDMARKS_IDXS['left_eye']
        (right_eye_beg, right_eye_end) = FACIAL_LANDMARKS_IDXS['right_eye']
        left_eye_points = shape[left_eye_beg:left_eye_end]
        right_eye_points = shape[right_eye_beg:right_eye_end]

        # calculate an angle between a line connecting centres of eyes
        left_eye_center = left_eye_points.mean(axis=0).astype('int')
        right_eye_center = right_eye_points.mean(axis=0).astype('int')
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_right_eye_x = 1.0 - self.desired_left_eye_pos[0]
        desired_dist = desired_right_eye_x - self.desired_left_eye_pos[0]
        desired_dist *= self.desired_face_photo_width
        scale = desired_dist / dist

        eyes_center = (int((left_eye_center[0] + right_eye_center[0]) // 2),
                       int((left_eye_center[1] + right_eye_center[1]) // 2))
        matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        tX = self.desired_face_photo_width * 0.5
        tY = self.desired_face_photo_height * self.desired_left_eye_pos[1]
        matrix[0, 2] += (tX - eyes_center[0])
        matrix[1, 2] += (tY - eyes_center[1])

        # apply the affine transformation
        (w, h) = (self.desired_face_photo_width, self.desired_face_photo_height)
        output = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC)

        return output
