import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.lstm.model import SmileAuthenticityPredictor
from models.lstm.data import prepare_data
from models._config import nns_config as nns_conf
from models.lstm.utils import get_trainer
from models.lstm.predictions import review_predictions
from config import FFS_COLS_NAMES


def train():
    data_module, test_data = prepare_data()
    model = SmileAuthenticityPredictor(num_features=len(FFS_COLS_NAMES), num_classes=nns_conf.num_classes)
    trainer = get_trainer()

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path='best')

    review_predictions(trainer=trainer, test_data=test_data)


if __name__ == '__main__':
    train()
