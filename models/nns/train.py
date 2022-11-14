import os
import sys
import torch
import time
import torch.nn as nn
import pickle as pkl
import matplotlib.pylab as plt
from copy import deepcopy
from datetime import timedelta
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchvision import models as mdls
from sklearn.model_selection import KFold
from statistics import fmean

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models._config import CNNLSTM_config as cnn_lstm_conf
from models._config import CNN3D_config as cnn_3d_conf
from models.nns.video_cnn_lstm import VideoCNNLSTM
from models.nns.videos_dataset import prepare_datasets
from models._config import nns_config as nns_conf
from config import NNS_WEIGHTS_DIR, NNS_PLOTS_DIR, NNS_LEARNING_DATA_DIR
from utils import get_current_time_str


def handle_arguments():
    arguments = sys.argv
    if len(arguments) != 2:
        raise Exception('Invalid number of parameters. This script accepts only ONE parameter - neural network '
                        'architecture number.')

    arguments[1] = int(arguments[1])

    if int(arguments[1]) not in [0, 1]:
        raise Exception('Invalid neural network architecture number.\n'
                        'Options to choose:\n'
                        '0: CNN + LSTM neural network\n'
                        '1: 3D CNN (18 layer Resnet3D) neural network')

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


def save_learning_process(model_name, training_proc_data, date_str):
    file_name = f'{model_name}-{date_str}.pkl'
    file_path = os.path.abspath(os.path.join(os.sep, NNS_LEARNING_DATA_DIR, file_name))
    with open(file_path, 'wb') as outp:
        pkl.dump(training_proc_data, outp)
    print(f'Training process data successfully saved into {file_name} file.')


def calculate_exec_time(start, end):
    exec_time = end - start
    return exec_time, str(timedelta(seconds=exec_time))


# def get_model_params(model_num, device):
#     training_proc_data = []
#     model_name, model, loss_func, optimizer, lr_scheduler, num_epochs = None, None, None, None, None, None
#
#     if model_num == 0:
#         model_name = 'CNN+LSTM_NN'
#
#         training_proc_data.append({
#             'model_name': model_name,
#             'dropout_p': cnn_lstm_conf.dropout_p,
#             'lstm_num_layers': cnn_lstm_conf.lstm_num_layers,
#             'lstm_hidden_size': cnn_lstm_conf.lstm_hidden_size,
#             'batch_size': cnn_lstm_conf.batch_size,
#             'learning_rate': cnn_lstm_conf.learning_rate,
#             'num_epochs': cnn_lstm_conf.num_epochs,
#         })
#
#         model = VideoCNNLSTM().to(device)
#         loss_func = nn.CrossEntropyLoss(reduction='sum')
#         optimizer = torch.optim.Adam(model.parameters(), lr=cnn_lstm_conf.learning_rate)
#         lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
#         num_epochs = cnn_lstm_conf.num_epochs
#
#     elif model_num == 1:
#         model_name = '3D_CNN_NN'
#
#         training_proc_data.append({
#             'model_name': model_name,
#             'batch_size': cnn_3d_conf.batch_size,
#             'learning_rate': cnn_3d_conf.learning_rate,
#             'num_epochs': cnn_3d_conf.num_epochs,
#         })
#
#         model = mdls.video.r3d_18(weights=mdls.video.R3D_18_Weights.DEFAULT, progress=False)
#         num_features = model.fc.in_features
#         model.fc = nn.Linear(num_features, nns_conf.num_classes)
#
#         loss_func = nn.CrossEntropyLoss(reduction='sum')
#         optimizer = torch.optim.SGD(model.parameters(), lr=cnn_3d_conf.learning_rate)
#         lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
#         num_epochs = cnn_3d_conf.num_epochs
#         if torch.cuda.is_available():
#             model.cuda()
#
#     return model_name, model, loss_func, optimizer, lr_scheduler, num_epochs
#
#
# def get_test_data_loader(test_data, model_num):
#     test_loader = None
#     if model_num == 0:
#         test_loader = DataLoader(dataset=test_data, batch_size=cnn_lstm_conf.batch_size, collate_fn=collate_fn_cnn_lstm)
#     elif model_num == 1:
#         test_loader = DataLoader(dataset=test_data, batch_size=cnn_3d_conf.batch_size, collate_fn=collate_fn_cnn3d)
#
#     return test_loader
#
#
# def get_train_val_data_loaders(train_val_data, train_ids, val_ids, model_num):
#     train_loader, val_loader = None, None
#     train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
#     val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)
#
#     if model_num == 0:
#         train_loader = DataLoader(dataset=train_val_data, batch_size=cnn_lstm_conf.batch_size, sampler=train_sampler,
#                                   collate_fn=collate_fn_cnn_lstm)
#         val_loader = DataLoader(dataset=train_val_data, batch_size=cnn_lstm_conf.batch_size, sampler=val_sampler,
#                                 collate_fn=collate_fn_cnn_lstm)
#     elif model_num == 1:
#         train_loader = DataLoader(dataset=train_val_data, batch_size=cnn_3d_conf.batch_size, sampler=train_sampler,
#                                   collate_fn=collate_fn_cnn3d)
#         val_loader = DataLoader(dataset=train_val_data, batch_size=cnn_3d_conf.batch_size, sampler=val_sampler,
#                                 collate_fn=collate_fn_cnn3d)
#
#     return train_loader, val_loader


def get_model_params(model_num, train_data, val_data, test_data, device):
    training_proc_data = []
    model_name, model, loss_func, optimizer, lr_scheduler, train_loader, val_loader, test_loader, num_epochs = \
        None, None, None, None, None, None, None, None, None

    if model_num == 0:
        model_name = 'CNN+LSTM_NN'

        training_proc_data.append({
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
        loss_func = nn.CrossEntropyLoss(reduction='sum')
        # optimizer = torch.optim.Adam(model.parameters(), lr=cnn_lstm_conf.learning_rate)
        optimizer = torch.optim.SGD(model.parameters(), lr=cnn_lstm_conf.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        num_epochs = cnn_lstm_conf.num_epochs

    elif model_num == 1:
        model_name = '3D_CNN_NN'

        training_proc_data.append({
            'model_name': model_name,
            'batch_size': cnn_3d_conf.batch_size,
            'learning_rate': cnn_3d_conf.learning_rate,
            'num_epochs': cnn_3d_conf.num_epochs,
        })

        train_loader = DataLoader(dataset=train_data, batch_size=cnn_3d_conf.batch_size, shuffle=True,
                                  collate_fn=collate_fn_cnn3d)
        val_loader = DataLoader(dataset=val_data, batch_size=cnn_3d_conf.batch_size, shuffle=False,
                                collate_fn=collate_fn_cnn3d)
        test_loader = DataLoader(dataset=test_data, batch_size=cnn_3d_conf.batch_size, shuffle=False,
                                 collate_fn=collate_fn_cnn3d)

        model = mdls.video.r3d_18(weights=mdls.video.R3D_18_Weights.DEFAULT, progress=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, nns_conf.num_classes)

        loss_func = nn.CrossEntropyLoss(reduction='sum')
        # optimizer = torch.optim.Adam(model.parameters(), lr=cnn_3d_conf.learning_rate)
        optimizer = torch.optim.SGD(model.parameters(), lr=cnn_3d_conf.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        num_epochs = cnn_3d_conf.num_epochs
        if torch.cuda.is_available():
            model.cuda()

    return model_name, model, loss_func, optimizer, lr_scheduler, train_loader, val_loader, test_loader, num_epochs


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
    model.load_state_dict(best_model_dict)  # test on the best model
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

    training_proc_data = []

    train_data, val_data, test_data = prepare_datasets()
    model_name, model, loss_func, optimizer, lr_scheduler, train_loader, val_loader, test_loader, num_epochs = \
        get_model_params(num_model, train_data, val_data, test_data, device)
    model_values_to_check = [model, loss_func, optimizer, lr_scheduler, train_loader, val_loader, test_loader]

    if all(mv is not None for mv in model_values_to_check):
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

        best_model_dict = deepcopy(model.state_dict())
        best_loss = float('inf')
        best_epoch = 1

        for epoch in range(1, num_epochs + 1):
            lr = get_lr(optimizer)

            # training
            train_loss, train_accuracy = train_loop(model, train_loader, optimizer, loss_func, device)
            # validating
            # val_loss, val_accuracy = val_loop(model, val_loader, optimizer, loss_func, device)
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
            'execution_time': exec_time,
            'best_val_loss': best_loss,
            'best_epoch': best_epoch
        })

        save_learning_process(model_name, training_proc_data, date_str)
        epochs_list = [i for i in range(1, num_epochs + 1)]

        return model_name, train_losses, val_losses, train_accuracies, val_accuracies, epochs_list

    else:
        print('Error: Some of the variables: model/loss function/optimizer/learning rate scheduler/train data loader/'
              'validation data loader/test data loader are None.\n')
        sys.exit(1)


# def train():
#     start = time.time()
#     _, num_model = handle_arguments()
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'Model will be trained on: {device}\n')
#
#     training_proc_data = []
#
#     train_data, val_data, test_data = prepare_datasets()
#     train_val_data = ConcatDataset([train_data, val_data])
#
#     model_name, model, loss_func, optimizer, lr_scheduler, num_epochs = get_model_params(num_model, device)
#     model_values_to_check = [model, loss_func, optimizer, lr_scheduler]
#
#     if all(mv is not None for mv in model_values_to_check):
#         all_train_losses, all_val_losses, all_train_accuracies, all_val_accuracies = [], [], [], []
#         folds_train_losses, folds_val_losses, folds_train_accuracies, folds_val_accuracies = [], [], [], []
#         k_fold = KFold(n_splits=nns_conf.k_folds, shuffle=True)
#         test_loader = get_test_data_loader(test_data, num_model)
#
#         best_model_dict = deepcopy(model.state_dict())
#         best_loss = float('inf')
#         best_epoch = 1
#
#         for num_fold, (train_ids, val_ids) in enumerate(k_fold.split(train_val_data)):
#             train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
#             print(f'##############################################\n'
#                   f'fold number: {num_fold+1}/{nns_conf.k_folds}')
#
#             train_loader, val_loader = get_train_val_data_loaders(train_val_data, train_ids, val_ids, num_model)
#
#             for epoch in range(1, num_epochs + 1):
#                 lr = get_lr(optimizer)
#
#                 # training
#                 train_loss, train_accuracy = train_loop(model, train_loader, optimizer, loss_func, device)
#                 # validating
#                 val_loss, val_accuracy = val_loop(model, val_loader, loss_func, device)
#
#                 if val_loss < best_loss:
#                     best_model_dict = deepcopy(model.state_dict())
#                     best_loss = val_loss
#                     best_epoch = epoch
#
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)
#                 train_accuracies.append(train_accuracy)
#                 val_accuracies.append(val_accuracy)
#
#                 all_train_losses.append(train_loss)
#                 all_val_losses.append(val_loss)
#                 all_train_accuracies.append(train_accuracy)
#                 all_val_accuracies.append(val_accuracy)
#
#                 print(f'----------------------------------------------\n'
#                       f'epoch: {epoch}/{num_epochs}\n'
#                       f'learning rate: {lr}\n\n'
#                       f'training loss: {train_loss}\n'
#                       f'validation loss: {val_loss}\n'
#                       f'training accuracy: {train_accuracy} %\n'
#                       f'validation accuracy: {val_accuracy} %\n')
#
#                 training_proc_data.append({
#                     'epoch': epoch,
#                     'learning_rate': lr,
#                     'training_loss': train_loss,
#                     'validation_loss': val_loss,
#                     'training_accuracy': train_accuracy,
#                     'validation_accuracy': val_accuracy
#                 })
#
#                 lr_scheduler.step(val_loss)
#
#             folds_train_losses.append(fmean(train_losses))
#             folds_val_losses.append(fmean(val_losses))
#             folds_train_accuracies.append(fmean(train_accuracies))
#             folds_val_accuracies.append(fmean(val_accuracies))
#
#         # testing
#         test_accuracy = test_loop(model, best_model_dict, test_loader, device)
#         print(f'test accuracy of the model ({model_name}): {test_accuracy} %')
#
#         save_best_model(model_name, best_model_dict)
#
#         end = time.time()
#         exec_time, str_exec_time = calculate_exec_time(start, end)
#
#         print(f'Learning process took: {str_exec_time}.\n'
#               f'best validation loss: {best_loss}\n'
#               f'best epoch: {best_epoch}')
#
#         training_proc_data.append({
#             'execution_time': exec_time,
#             'best_val_loss': best_loss,
#             'best_epoch': best_epoch
#         })
#
#         save_learning_process(model_name, training_proc_data)
#         epochs_list = [i for i in range(1, num_epochs * nns_conf.k_folds + 1)]
#
#         return model_name, all_train_losses, all_val_losses, all_train_accuracies, all_val_accuracies, epochs_list
#
#     else:
#         print('Error: Some of the variables: model/loss function/optimizer/learning rate scheduler/train data loader/'
#               'validation data loader/test data loader are None.\n')
#         sys.exit(1)


def plot_training_values(model_name, x_values, train_values, val_values, train_label, val_label, label, date_str, loc):
    plt.plot(x_values, train_values, label=train_label)
    plt.plot(x_values, val_values, label=val_label)
    plt.legend(loc=loc)
    plt.xticks(x_values)
    plt.xlabel('epochs')
    plt.ylabel(label)
    plt.title(f'{label.capitalize()} in subsequent epochs for learning {model_name}')

    plot_file_title = f'{label}_dur_tr_plot-{model_name}-{date_str}.png'
    plt.savefig(os.path.abspath(os.path.join(os.sep, NNS_PLOTS_DIR, plot_file_title)))
    plt.show()


if __name__ == '__main__':
    date_str = get_current_time_str()
    mn, tl, vl, ta, va, el = train(date_str)
    plot_training_values(mn, x_values=el, train_values=tl, val_values=vl, train_label='training dataset',
                         val_label='validation dataset', label='loss', date_str=date_str, loc='lower left')
    plot_training_values(mn, x_values=el, train_values=ta, val_values=va, train_label='training dataset',
                         val_label='validation dataset', label='accuracy', date_str=date_str, loc='upper left')
