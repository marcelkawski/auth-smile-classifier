import numpy as np

batch_size = 64
test_size = 0.3
video_frame_size = (144, 256)
means = np.array([0.485, 0.456, 0.406])
stds = np.array([0.229, 0.224, 0.225])
