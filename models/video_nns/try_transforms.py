import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.models_config import CNNLSTM_imgs_transforms_config as cnn_lstm_itc
from models.models_config import CNN3D_imgs_transforms_config as cnn_3d_itc


def perform_transform(num_model, transf):
    orig_img = Image.open(r'C:\Users\Marcel\Studia\mgr\praca_magisterska\auth-smile-classifier\data\faces\013_deliberate_smile_1.mp4\013_deliberate_smile_1.mp4_frame0.jpg')

    # orig_img_to_save = T.ToTensor()(orig_img)
    # save_image(orig_img_to_save, r'C:\Users\Marcel\Studia\mgr\praca_magisterska\obrazki\transforms\oryginal.jpg')

    if num_model == 0:
        transforms_dict = cnn_lstm_itc
    elif num_model == 1:
        transforms_dict = cnn_3d_itc

    if transf == 'resize':
        img_after_transform = T.Resize(size=(transforms_dict.h, transforms_dict.w))(orig_img)
    elif transf == 'normalize':
        img_after_transform = T.Resize(size=(transforms_dict.h, transforms_dict.w))(orig_img)
        img_after_transform = T.ToTensor()(img_after_transform)
        img_after_transform = T.Normalize(transforms_dict.means, transforms_dict.stds)(img_after_transform)
        img_after_transform = T.ToPILImage()(img_after_transform)

    img_to_save = T.ToTensor()(img_after_transform)
    save_image(img_to_save, rf'C:\Users\Marcel\Studia\mgr\praca_magisterska\obrazki\transforms\{transf}{num_model}.jpg')



if __name__ == '__main__':
    perform_transform(1, 'resize')
