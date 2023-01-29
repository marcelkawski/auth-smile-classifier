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
    'h': 224,
    'w': 224
})

CNNLSTM_config = DotMap({
    # 'dropout_p': 0.5,
    'dropout_p': 0.1,
    'lstm_num_layers': 1,
    'lstm_hidden_size': 100,
    # 'lstm_num_layers': 3,
    # 'lstm_hidden_size': 256,
    'batch_size': 1,
    # 'batch_size': 4,
    # 'learning_rate': 1e-6,
    'learning_rate': 3e-5,
    # 'learning_rate': 1e-4,
    'num_epochs': 20,
})

CNN3D_imgs_transforms_config = DotMap({
    'h': 112,
    'w': 112
})

CNN3D_config = DotMap({
    'batch_size': 1,
    # 'batch_size': 4,
    # 'learning_rate': 1e-6,
    'learning_rate': 3e-5,
    # 'learning_rate': 1e-4,
    'num_epochs': 20,
})

LSTM_config = DotMap({
    'num_epochs': 300,
    'batch_size': 64,
    'num_hidden': 256,
    'num_lstm_layers': 3,
    'dropout': 0.75,
    'learning_rate': 0.0001
})

# LSTM_config = DotMap({
#     'num_epochs': 600,
#     'batch_size': 64,
#     'num_hidden': 512,
#     'num_lstm_layers': 3,
#     'dropout': 0.75,
#     'learning_rate': 0.00001
# })

# LSTM_config = DotMap({
#     'num_epochs': 800,
#     'batch_size': 128,
#     'num_hidden': 768,
#     'num_lstm_layers': 4,
#     'dropout': 0.75,
#     'learning_rate': 0.00001
# })
