# import cv2
#
#
# if __name__ == "__main__":
#     face_cascade = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')
#
#     path = r'C:\Users\Marcel\Studia\mgr\praca_magisterska\auth-smile-classifier\data\frames\001_deliberate_smile_2.mp4\001_deliberate_smile_2.mp4_frame0.jpg'
#     img_number = 0
#
#     img = cv2.imread(path, 1)  # zobaczyc bez 1
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#
#     for x, y, w, h in faces:
#         roi_color = img[y:y + h, x:x + w]
#     resized = cv2.resize(roi_color, (128, 128))
#     cv2.imwrite(rf'faces\{str(img_number)}.jpg', resized)
#
#     img_number += 1
