import os
import sys
import torch
import time
import torch.nn as nn
import pickle as pkl
import matplotlib.pylab as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models._config import RNN_config as rnn_conf
from models.nns.video_cnn_lstm import VideoCNNLSTM
from models.nns.dataset import prepare_datasets
from config import NNS_WEIGHTS_DIR, NNS_PLOTS_DIR
from utils import get_current_time_str


def handle_arguments():
    arguments = sys.argv
    if len(arguments) != 2:
        raise Exception('Invalid number of parameters. This script accepts only ONE parameter - neural network '
                        'architecture number.')

    arguments[1] = int(arguments[1])

    if int(arguments[1]) not in [0]:
        raise Exception('Invalid neural network architecture number.\n'
                        'Options to choose:\n'
                        '0: CNN + LSTM neural network')

    return arguments


def collate_fn_cnn_lstm(batch):
    frames_batch, auth_batch = list(zip(*batch))
    frames_batch = [frames for frames in frames_batch if len(frames) > 0]
    auth_batch = [torch.tensor(l) for l, frames in zip(auth_batch, frames_batch) if len(frames) > 0]
    frames_tensor = torch.stack(frames_batch)
    auth_tensor = torch.stack(auth_batch)
    return frames_tensor, auth_tensor


def train():
    train_data, val_data, test_data = prepare_datasets()

    _, model = handle_arguments()
    epochs_results = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Model will be trained on: {device}\n')

    if model == 0:
        model_name = 'CNN+LSTM_NN'

        train_loader = DataLoader(dataset=train_data, batch_size=rnn_conf.batch_size, shuffle=True,
                                  collate_fn=collate_fn_cnn_lstm)
        val_loader = DataLoader(dataset=val_data, batch_size=rnn_conf.batch_size, shuffle=False,
                                collate_fn=collate_fn_cnn_lstm)
        test_loader = DataLoader(dataset=test_data, batch_size=rnn_conf.batch_size, shuffle=False,
                                 collate_fn=collate_fn_cnn_lstm)

        model = VideoCNNLSTM().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=rnn_conf.learning_rate)

        train_losses = []
        val_losses = []
    
        for epoch in range(1, rnn_conf.num_epochs + 1):
            train_loss, val_loss = 0.0, 0.0
    
            model.train()
            for frames, auth in train_loader:
                frames = frames.to(device)
                auth = auth.to(device)

                optimizer.zero_grad()
                output = model(frames)

                loss = criterion(output, auth)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * frames.size(0)
    
            model.eval()
            for frames, auth in val_loader:
                frames = frames.to(device)
                auth = auth.to(device)

                output = model(frames)
                loss = criterion(output, auth)
                val_loss += loss.item() * frames.size(0)

            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            epochs_results.append((epoch, train_loss, val_loss))

            print(f'----------------------------------------------\n'
                  f'epoch: {epoch}/{rnn_conf.num_epochs}\n\n'
                  f'training loss: {train_loss}\n'
                  f'validation loss: {val_loss}\n')

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for frames, auths in test_loader:
                frames = frames.to(device)
                auths = auths.to(device)

                outputs = model(frames)
                _, predicted = torch.max(outputs.data, 1)

                total += auths.size(0)
                correct += (predicted == auths).sum().item()

            print(f'test accuracy of the model ({model_name}): {100 * correct / total} %')

        weights_file_title = f'weights-{model_name}-{get_current_time_str()}'
        torch.save(model.state_dict(), os.path.abspath(os.path.join(os.sep, NNS_WEIGHTS_DIR, weights_file_title)))

        return model_name, train_losses, val_losses


def plot_training(model_name, train_losses, val_losses):
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(f'Loss in subsequent epochs for learning {model_name}')

    plot_file_tile = f'loss_dur_tr_plot-{model_name}-{get_current_time_str()}.png'
    plt.savefig(os.path.abspath(os.path.join(os.sep, NNS_PLOTS_DIR, plot_file_tile)))
    plt.show()


if __name__ == '__main__':
    mn, tl, vl = train()
    plot_training(mn, tl, vl)
