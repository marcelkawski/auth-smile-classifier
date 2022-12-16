import sys
import os
from dotmap import DotMap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


nns_config = DotMap({
    'test_size': 0.2,
    'val_size': 0.2,  # <- <this_value> * <train_size>
    'num_classes': 2,
    # 'k_folds': 5
})

CNNLSTM_imgs_transforms_config = DotMap({
    'means': [0.485, 0.456, 0.406],
    'stds': [0.229, 0.224, 0.225],
    'h': 224,
    'w': 224
})

CNNLSTM_config = DotMap({
    'dropout_p': 0.5,
    'lstm_num_layers': 1,
    'lstm_hidden_size': 100,
    # 'lstm_num_layers': 3,
    # 'lstm_hidden_size': 256,
    # 'batch_size': 1,
    'batch_size': 4,
    'learning_rate': 1e-6,
    # 'learning_rate': 3e-5,
    # 'learning_rate': 1e-4,
    'num_epochs': 20,
})

CNN3D_imgs_transforms_config = DotMap({
    'means': [0.43216, 0.394666, 0.37645],
    'stds': [0.22803, 0.22145, 0.216989],
    'h': 112,
    'w': 112
})

CNN3D_config = DotMap({
    'batch_size': 4,
    # 'learning_rate': 1e-6,
    'learning_rate': 3e-5,
    # 'learning_rate': 1e-4,
    'num_epochs': 20,
})

LSTM_config = DotMap({
    'num_epochs': 500,
    'batch_size': 64,
    'num_hidden': 256,
    'num_lstm_layers': 3,
    'dropout': 0.75,
    'learning_rate': 0.0001
})
