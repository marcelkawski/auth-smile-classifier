import sys
import os
from dotmap import DotMap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CURRENT_MIN_NUM_SMILE_FRAMES


nns_config = DotMap({
    'test_size': 0.2,
    'val_size': 0.2,  # <- <this_value> * <train_size>
    'num_classes': 2,
    # 'k_folds': 5
})

#########################################################
# CNN LSTM config

CNNLSTM_imgs_transforms_config = DotMap({
    'means': [0.485, 0.456, 0.406],
    'stds': [0.229, 0.224, 0.225],
    'h': 224,
    'w': 224
})

CNNLSTM_config = DotMap({
    'dropout_p': 0.1,
    'lstm_num_layers': 1,
    'lstm_hidden_size': 100,
    'batch_size': 2,
    # 'learning_rate': 3e-5,
    'learning_rate': 1e-4,
    'num_epochs': 20,
})

#########################################################
# 3D CNN config

CNN3D_imgs_transforms_config = DotMap({
    'means': [0.43216, 0.394666, 0.37645],
    'stds': [0.22803, 0.22145, 0.216989],
    'h': 112,
    'w': 112
})

CNN3D_config = DotMap({
    'batch_size': 1,
    'learning_rate': 3e-5,
    # 'learning_rate': 1e-4,
    'num_epochs': 5,
})

#########################################################
# LSTM config

# LSTM_config = DotMap({
#     'learning_rate': 1e-4,
#     'batch_size': 10,
#     'num_epochs': 500,
#     'num_features': 4,
#     'seq_length': CURRENT_MIN_NUM_SMILE_FRAMES,
#     'num_hidden': 20,
#     'num_lstm_layers': 1
# })

LSTM_config = DotMap({
    'num_epochs': 300,
    'batch_size': 64,
    'num_hidden': 256,
    'num_lstm_layers': 3,
    'dropout': 0.75,
    'learning_rate': 0.0001
})
