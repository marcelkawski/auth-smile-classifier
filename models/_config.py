from dotmap import DotMap


nns_config = DotMap({
    'test_size': 0.2,
    'val_size': 0.2,  # <- <this_value> * <train_size>
    'num_classes': 2
})

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
    'learning_rate': 3e-5,
    'num_epochs': 20,
})

CNN3D_imgs_transforms_config = DotMap({
    'means': [0.43216, 0.394666, 0.37645],
    'stds': [0.22803, 0.22145, 0.216989],
    'h': 112,
    'w': 112
})

CNN3D_config = DotMap({
    'batch_size': 2,
    'learning_rate': 3e-5,
    'num_epochs': 20,
})
