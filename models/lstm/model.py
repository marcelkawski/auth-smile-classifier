import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.functional import accuracy


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models._config import LSTM_config as lstm_conf


class FacesFeaturesLSTM(nn.Module):
    def __init__(self, num_features, num_classes, num_hidden=lstm_conf.num_hidden,
                 num_lstm_layers=lstm_conf.num_lstm_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=num_hidden,  # number of neurons in each layer
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_conf.dropout
        )
        self.classifier = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()  # for multi GPU purposes
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]
        return self.classifier(out)


class SmileAuthenticityPredictor(pl.LightningModule):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.model = FacesFeaturesLSTM(num_features=num_features, num_classes=num_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, auths=None):
        output = self.model(x)
        loss = 0
        if auths is not None:
            loss = self.loss_func(output, auths)
        return loss, output

    def training_step(self, batch, batch_idx):
        faces_features = batch['faces_features']
        authenticities = batch['authenticity']

        loss, outputs = self(faces_features, authenticities)
        predictions = torch.argmax(outputs, dim=1)
        acc = accuracy(predictions, authenticities)

        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_accuracy', acc, prog_bar=True, logger=True)

        return {
            'loss': loss,
            'accuracy': accuracy
        }

    def validation_step(self, batch, batch_idx):
        faces_features = batch['faces_features']
        authenticities = batch['authenticity']

        loss, outputs = self(faces_features, authenticities)
        predictions = torch.argmax(outputs, dim=1)
        acc = accuracy(predictions, authenticities)

        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_accuracy', acc, prog_bar=True, logger=True)

        return {
            'loss': loss,
            'accuracy': accuracy
        }

    def test_step(self, batch, batch_idx):
        faces_features = batch['faces_features']
        authenticities = batch['authenticity']

        loss, outputs = self(faces_features, authenticities)
        predictions = torch.argmax(outputs, dim=1)
        acc = accuracy(predictions, authenticities)

        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_accuracy', acc, prog_bar=True, logger=True)

        return {
            'loss': loss,
            'accuracy': accuracy
        }

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=lstm_conf.learning_rate)
