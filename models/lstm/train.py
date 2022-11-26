import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.lstm.model import SmileAuthenticityPredictor
from models.lstm.data import prepare_data
from models._config import nns_config as nns_conf
from models._config import LSTM_config as lstm_conf


def train():
    data_module, ffs_columns = prepare_data()

    model = SmileAuthenticityPredictor(num_features=len(ffs_columns), num_classes=nns_conf.num_classes)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-checkpoint',
        save_top_k=1,  # only the best model
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    logger = TensorBoardLogger('lightning_logs', name='authenticity')

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=lstm_conf.num_epochs,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=10
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


if __name__ == '__main__':
    train()
