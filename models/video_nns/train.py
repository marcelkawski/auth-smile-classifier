import os
import sys
import torch
import time
import torch.nn as nn
import matplotlib.pylab as plt
import numpy as np
from copy import deepcopy
from datetime import timedelta
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models as mdls

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.models_config import nns_config as nns_conf
from models.models_config import CNNLSTM_config as cnn_lstm_conf
from models.models_config import CNN3D_config as cnn_3d_conf
from models.models_config import CNNLSTM_imgs_transforms_config as cnn_lstm_itc
from models.models_config import CNN3D_imgs_transforms_config as cnn_3d_itc
from models.video_nns.video_cnn_lstm import VideoCNNLSTM
from models.video_nns.dataset import prepare_datasets
from config import NNS_WEIGHTS_DIR, NNS_PLOTS_DIR, NNS_LEARNING_DATA_DIR
from utils import get_current_time_str
from data_prep.data_prep_utils import save_dict_to_json_file


def handle_arguments():
    arguments = sys.argv
    if len(arguments) != 2:
        raise Exception('Invalid number of parameters. This script accepts only ONE parameter - neural network '
                        'architecture number.\n')

    arguments[1] = int(arguments[1])

    if arguments[1] not in [0, 1]:
        raise Exception('Invalid neural network architecture number.\n'
                        'Options to choose:\n'
                        '--------------------------------------------------\n'
                        'Analyzing videos:\n'
                        '0: CNN + LSTM neural network\n'
                        '1: 3D CNN (18 layer Resnet3D) neural network\n')

    return arguments


def collate_fn_cnn_lstm(batch):
    frames_batch, auth_batch = list(zip(*batch))
    frames_batch = [frames for frames in frames_batch if len(frames) > 0]
    auth_batch = [torch.tensor(l) for l, frames in zip(auth_batch, frames_batch) if len(frames) > 0]
    frames_tensor = torch.stack(frames_batch)
    auth_tensor = torch.stack(auth_batch)
    return frames_tensor, auth_tensor


def collate_fn_cnn3d(batch):
    frames_batch, auth_batch = list(zip(*batch))
    frames_batch = [frames for frames in frames_batch if len(frames) > 0]
    auth_batch = [torch.tensor(l) for l, frames in zip(auth_batch, frames_batch) if len(frames) > 0]
    frames_tensor = torch.stack(frames_batch)
    frames_tensor = torch.transpose(frames_tensor, 2, 1)
    labels_tensor = torch.stack(auth_batch)
    return frames_tensor, labels_tensor


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_learning_process(model_name, training_proc_data, _date_str):
    file_name = f'{model_name}-{_date_str}.json'
    save_dict_to_json_file(NNS_LEARNING_DATA_DIR, file_name, training_proc_data, time_str=_date_str)
    print(f'Training process data successfully saved into {file_name} file.')


def calculate_exec_time(start, end):
    exec_time = end - start
    return exec_time, str(timedelta(seconds=exec_time))


def get_model_params(model_num, train_data, val_data, test_data, device):
    training_proc_data = []
    model_name, model, loss_func, optimizer, lr_scheduler, train_loader, val_loader, test_loader, num_epochs, \
        collate_fn, transforms_dict, config_dict = None, None, None, None, None, None, None, None, None, None, {}, {}

    if model_num == 0:
        model_name = 'CNN+LSTM_NN'
        transforms_dict = cnn_lstm_itc
        config_dict = cnn_lstm_conf
        collate_fn = collate_fn_cnn_lstm

        model = VideoCNNLSTM().to(device)

    elif model_num == 1:
        model_name = '3D_CNN_NN'
        transforms_dict = cnn_3d_itc
        config_dict = cnn_3d_conf
        collate_fn = collate_fn_cnn3d

        model = mdls.video.r3d_18(weights=mdls.video.R3D_18_Weights.DEFAULT, progress=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, nns_conf.num_classes)

        if torch.cuda.is_available():
            model.cuda()

    loss_func = nn.CrossEntropyLoss(reduction='sum')
    if config_dict.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config_dict.learning_rate)
    elif config_dict.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cnn_lstm_conf.learning_rate)
    else:
        raise Exception('Incorrect optimizer given in the model configuration file. Options to choose:\n'
                        '- "SGD" (stochastic gradient descent)\n'
                        '- "Adam"\n')
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    num_epochs = config_dict.num_epochs

    if collate_fn is not None:
        train_loader = DataLoader(dataset=train_data, batch_size=config_dict.batch_size, shuffle=True,
                                  collate_fn=collate_fn)
        val_loader = DataLoader(dataset=val_data, batch_size=config_dict.batch_size, shuffle=False,
                                collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test_data, batch_size=config_dict.batch_size, shuffle=False,
                                 collate_fn=collate_fn)
    else:
        train_loader = DataLoader(dataset=train_data, batch_size=config_dict.batch_size, shuffle=False)
        val_loader = DataLoader(dataset=val_data, batch_size=config_dict.batch_size, drop_last=True, shuffle=False)
        test_loader = DataLoader(dataset=test_data, batch_size=config_dict.batch_size, drop_last=True,
                                 shuffle=False)

    training_proc_data.append({
        'model_name': model_name,
        'config': {
            'nns_config': dict(nns_conf),
            'transforms': dict(transforms_dict),
            'specific_config': dict(config_dict)
        },
        'loss_func': type(loss_func).__name__,
        'optimizer': type(optimizer).__name__,
        'lr_scheduler': type(lr_scheduler).__name__
    })

    return model_name, model, loss_func, optimizer, lr_scheduler, train_loader, val_loader, test_loader, num_epochs, \
           training_proc_data


def train_loop(model, train_loader, optimizer, loss_func, device):
    model.train()
    correct, total = 0, 0
    train_loss = 0.0
    for frames, auth in train_loader:
        frames = frames.to(device)
        auth = auth.to(device)

        optimizer.zero_grad()
        output = model(frames)
        _, predicted = torch.max(output.data, 1)

        total += auth.size(0)
        correct += (predicted == auth).sum().item()

        loss = loss_func(output, auth)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * frames.size(0)

    train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy


def val_loop(model, val_loader, loss_func, device):
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0
    for frames, auth in val_loader:
        frames = frames.to(device)
        auth = auth.to(device)

        output = model(frames)
        _, predicted = torch.max(output.data, 1)

        total += auth.size(0)
        correct += (predicted == auth).sum().item()

        loss = loss_func(output, auth)
        val_loss += loss.item() * frames.size(0)

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy


def test_loop(model, best_model_dict, test_loader, device):
    model.eval()
    model.load_state_dict(best_model_dict)  # tests on the best model
    with torch.no_grad():
        correct, total = 0, 0
        for frames, auths in test_loader:
            frames = frames.to(device)
            auths = auths.to(device)

            outputs = model(frames)
            _, predicted = torch.max(outputs.data, 1)

            total += auths.size(0)
            correct += (predicted == auths).sum().item()

    test_accuracy = 100 * correct / total
    return test_accuracy


def save_best_model(model_name, best_model_dict, date_str):
    weights_file_title = f'weights-{model_name}-{date_str}.pt'
    torch.save(best_model_dict, os.path.abspath(os.path.join(os.sep, NNS_WEIGHTS_DIR, weights_file_title)))


def train(date_str):
    start = time.time()
    _, num_model = handle_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Model will be trained on: {device}\n')

    train_data, val_data, test_data = prepare_datasets(num_model)

    model_name, model, loss_func, optimizer, lr_scheduler, train_loader, val_loader, test_loader, num_epochs, \
        training_proc_data = get_model_params(num_model, train_data, val_data, test_data, device)
    model_values_to_check = [model, loss_func, optimizer, train_loader, val_loader, test_loader]

    if all(mv is not None for mv in model_values_to_check):
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

        best_model_dict = deepcopy(model.state_dict())
        best_loss = float('inf')
        best_epoch = 1

        print('epochs: ', num_epochs)

        for epoch in range(1, num_epochs + 1):
            lr = get_lr(optimizer)

            # training
            train_loss, train_accuracy = train_loop(model, train_loader, optimizer, loss_func, device)
            # validating
            val_loss, val_accuracy = val_loop(model, val_loader, loss_func, device)

            if val_loss < best_loss:
                best_model_dict = deepcopy(model.state_dict())
                best_loss = val_loss
                best_epoch = epoch

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            print(f'----------------------------------------------\n'
                  f'epoch: {epoch}/{num_epochs}\n'
                  f'learning rate: {lr}\n\n'
                  f'training loss: {train_loss}\n'
                  f'validation loss: {val_loss}\n'
                  f'training accuracy: {train_accuracy} %\n'
                  f'validation accuracy: {val_accuracy} %\n')

            training_proc_data.append({
                'epoch': epoch,
                'learning_rate': lr,
                'training_loss': train_loss,
                'validation_loss': val_loss,
                'training_accuracy': train_accuracy,
                'validation_accuracy': val_accuracy
            })

            if lr_scheduler is not None:
                lr_scheduler.step(val_loss)

        # testing
        test_accuracy = test_loop(model, best_model_dict, test_loader, device)
        print(f'test accuracy of the model ({model_name}): {test_accuracy} %')

        save_best_model(model_name, best_model_dict, date_str)

        end = time.time()
        exec_time, str_exec_time = calculate_exec_time(start, end)

        print(f'Learning process took: {str_exec_time}.\n'
              f'best validation loss: {best_loss}\n'
              f'best epoch: {best_epoch}')

        training_proc_data.append({
            'execution_time': str_exec_time,
            'best_val_loss': best_loss,
            'best_epoch': best_epoch,
            'test_accuracy': test_accuracy
        })

        save_learning_process(model_name, training_proc_data, date_str)
        epochs_list = [i for i in range(num_epochs)]
        if num_epochs < 100:
            step = 1
        else:
            step = 100

        return model_name, train_losses, val_losses, train_accuracies, val_accuracies, epochs_list, step

    else:
        raise Exception('Error: Some of the variables: model/loss function/optimizer/learning rate scheduler/train '
                        'data loader/validation data loader/tests data loader are None.\n')


def plot_training_values(model_name, x_values, train_values, val_values, train_label, val_label, label, date_str, loc,
                         step):
    plt.plot(x_values, train_values, label=train_label)
    plt.plot(x_values, val_values, label=val_label)
    plt.legend(loc=loc)
    plt.xticks(x_values)
    plt.xticks(np.arange(x_values[0], x_values[-1], step=step))
    plt.xlabel('epochs')
    plt.ylabel(label)
    plt.title(f'{label.capitalize()} in subsequent epochs for learning {model_name}')

    plot_file_title = f'{label}_dur_tr_plot-{model_name}-{date_str}.png'
    plt.savefig(os.path.abspath(os.path.join(os.sep, NNS_PLOTS_DIR, plot_file_title)))
    plt.show()


if __name__ == '__main__':
    d_str = get_current_time_str()
    mn, tl, vl, ta, va, el, st = train(d_str)
    plot_training_values(mn, x_values=el, train_values=tl, val_values=vl, train_label='training dataset',
                         val_label='validation dataset', label='loss', date_str=d_str, loc='lower left', step=st)
    plot_training_values(mn, x_values=el, train_values=ta, val_values=va, train_label='training dataset',
                         val_label='validation dataset', label='accuracy', date_str=d_str, loc='upper left',
                         step=st)
