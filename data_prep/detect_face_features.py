# import os
# import sys
# import cv2
# import dlib
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# from config import FACES_FEATURES_DET_FP
#
#
# if __name__ == '__main__':
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor(FACES_FEATURES_DET_FP)
#
#     # read the image
#     img = cv2.imread(r'C:\Users\Marcel\Studia\mgr\praca_magisterska\auth-smile-classifier\data\aligned_faces\001_deliberate_smile_2.mp4\001_deliberate_smile_2.mp4_frame101.jpg')
#
#     # Convert image into grayscale
#     gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
#
#     # Use detector to find landmarks
#     faces = detector(gray)
#     for face in faces:
#         x1 = face.left() # left point
#         y1 = face.top() # top point
#         x2 = face.right() # right point
#         y2 = face.bottom() # bottom point
#
#         # Create landmark object
#         landmarks = predictor(image=gray, box=face)
#
#         # Loop through all the points
#         for n in range(0, 68):
#             x = landmarks.part(n).x
#             y = landmarks.part(n).y
#
#             # Draw a circle
#             cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
#
#     winname = "Face"
#     cv2.namedWindow(winname)
#     cv2.moveWindow(winname, 40, 30)
#     cv2.imshow(winname=winname, mat=img)
#     cv2.waitKey(delay=0)
#     cv2.destroyAllWindows()
