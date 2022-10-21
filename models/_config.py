import numpy as np
from dotmap import DotMap

nns_config = DotMap({
    'test_size': 0.2,
    'val_size': 0.2,  # <- <this_value> * <train_size>
    'num_classes': 2
})

imgs_transforms_config = DotMap({
    'means': np.array([0.485, 0.456, 0.406]),
    'stds': np.array([0.229, 0.224, 0.225])
})

video_CNN1_config = DotMap({
    'num_epochs': 35,
    'num_classes': 2,
    'batch_size': 25,
    'learning_rate': 0.001,
    'kernel_size': 3
})


