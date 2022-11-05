import os
import sys
import torch
import time
import torch.nn as nn
import pickle as pkl
import matplotlib.pylab as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models._config import CNNLSTM_config as cnn_lstm_conf
from models.nns.video_cnn_lstm import VideoCNNLSTM
from models.nns.dataset import prepare_datasets
from config import NNS_WEIGHTS_DIR, NNS_PLOTS_DIR, NNS_LEARNING_DATA_DIR
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


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_learning_process(model_name, training_data):
    date_str = get_current_time_str()
    file_name = f'{model_name}-{date_str}.pkl'
    file_path = os.path.abspath(os.path.join(os.sep, NNS_LEARNING_DATA_DIR, file_name))
    with open(file_path, 'wb') as outp:
        pkl.dump(training_data, outp)
    print(f'Training process data successfully saved into {file_name} file.')


def train():
    train_data, val_data, test_data = prepare_datasets()

    _, model = handle_arguments()
    epochs_results = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Model will be trained on: {device}\n')

    training_data = []

    if model == 0:
        model_name = 'CNN+LSTM_NN'

        training_data.append({
            'model_name': model_name,
            'dropout_p': cnn_lstm_conf.dropout_p,
            'lstm_num_layers': cnn_lstm_conf.lstm_num_layers,
            'lstm_hidden_size': cnn_lstm_conf.lstm_hidden_size,
            'batch_size': cnn_lstm_conf.batch_size,
            'learning_rate': cnn_lstm_conf.learning_rate,
            'num_epochs': cnn_lstm_conf.num_epochs,
        })

        train_loader = DataLoader(dataset=train_data, batch_size=cnn_lstm_conf.batch_size, shuffle=True,
                                  collate_fn=collate_fn_cnn_lstm)
        val_loader = DataLoader(dataset=val_data, batch_size=cnn_lstm_conf.batch_size, shuffle=False,
                                collate_fn=collate_fn_cnn_lstm)
        test_loader = DataLoader(dataset=test_data, batch_size=cnn_lstm_conf.batch_size, shuffle=False,
                                 collate_fn=collate_fn_cnn_lstm)

        model = VideoCNNLSTM().to(device)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=cnn_lstm_conf.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        train_losses = []
        val_losses = []
    
        for epoch in range(1, cnn_lstm_conf.num_epochs + 1):
            train_loss, val_loss = 0.0, 0.0
            lr = get_lr(optimizer)
    
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
                  f'epoch: {epoch}/{cnn_lstm_conf.num_epochs}\n'
                  f'learning rate: {lr}\n\n'
                  f'training loss: {train_loss}\n'
                  f'validation loss: {val_loss}\n')

            training_data.append({
                'epoch': epoch,
                'learning_rate': lr,
                'training_loss': train_loss,
                'validation_loss': val_loss
            })

            lr_scheduler.step(val_loss)

        save_learning_process(model_name, training_data)

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
