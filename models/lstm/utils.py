import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models._config import LSTM_config as lstm_conf


def get_trainer():
    logger = TensorBoardLogger('lightning_logs', name='authenticity')
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-checkpoint',
        save_top_k=1,  # only the best model
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=lstm_conf.num_epochs,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=10
    )

    return trainer
