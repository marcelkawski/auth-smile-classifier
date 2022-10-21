import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models._config import video_CNN1_config as vc1c
from models.nns.video_cnn import VideoCNN
from models.nns.dataset import prepare_datasets


def handle_arguments():
    arguments = sys.argv
    if len(arguments) != 2:
        raise Exception('Invalid number of parameters. This script accepts only ONE parameter - neural network '
                        'architecture number.')

    arguments[1] = int(arguments[1])

    if int(arguments[1]) not in [0]:
        raise Exception('Invalid neural network architecture number.\n'
                        'Options to choose:\n'
                        '0: Convolutional Neural Network (CNN)')

    return arguments


def train():
    train_data, val_data, test_data = prepare_datasets()

    _, model = handle_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Model will be trained on: {device}\n')

    if model == 0:
        train_loader = DataLoader(dataset=train_data, batch_size=vc1c.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset=val_data, batch_size=vc1c.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(dataset=test_data, batch_size=vc1c.batch_size, shuffle=False, num_workers=0)

        model = VideoCNN().to(device)
        print(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=vc1c.learning_rate)

        train_losses, val_losses = [], []

        for epoch in range(vc1c.num_epochs):
            train_loss, val_loss = 0.0, 0.0

            model.train()
            for frames, authenticity in train_loader:
                frames = frames.to(device)
                authenticity = authenticity.to(device)

                optimizer.zero_grad()
                output = model(frames)
                loss = criterion(output, authenticity)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * frames.size(0)

            model.eval()
            for frames, authenticity in train_loader:
                frames = frames.to(device)
                authenticity = authenticity.to(device)

                output = model(frames)
                loss = criterion(output, authenticity)
                val_loss += loss.item() * frames.size(0)

            # Calculate average losses.
            train_loss = train_loss / len(train_loader.sampler)
            valid_loss = val_loss / len(val_loader.sampler)
            train_losses.append(train_loss)
            val_losses.append(valid_loss)


if __name__ == '__main__':
    train()
