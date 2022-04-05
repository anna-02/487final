import time
import copy
import math
from typing import List
from unicodedata import bidirectional
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import get_max_lengths, get_evaluation
from src.dataset import MyDataset
from src.final_fnn_model import FFNN
from tensorboardX import SummaryWriter
import argparse
import shutil
import itertools
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optimizer
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def get_loss_fn():
    """
    Return the loss function you will use to train the model.
    nn.CrossEntropyLoss()
    Hint: nn.BCEWithLogitsLoss
    """
    return nn.CrossEntropyLoss()


def calculate_loss(logits, labels, loss_fn):
    """
    Calculate the loss.
    Input:
        - logits: output logits from the model, 1-d tensor
        - labels: true label.
        - loss_fn: loss function

    """
    return loss_fn(logits, labels)


def get_optimizer(net, lr, weight_decay):
    """
    Return the optimizer (Adam) you will use to train the model.
    Input:
        - net: model
        - lr: initial learning_rate
        - weight_decay: weight_decay in optimizer
    """
    return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)


def get_hyper_parameters():
    """
    Return a list of hyper parameters that you will search.
    Return:
        - hidden_dim: dimension for hidden layer
        - lr: learning_rates
        - weight_decay: weight_decays

    Note: it takes about 1~2 minutes for one set of hyper parameters on Google Colab with GPU
    """
    hidden_dim, lr, weight_decay = [], [], []

    ######## TODO: try different lr, weight_decay, hidden_dim ##########
    hidden_dim = [50]
    lr = [1e-3, 1e-4]
    weight_decay = [1e-5, 1e-4]
    ######################### End of your code #########################

    return hidden_dim, lr, weight_decay


def train_model(net, training_generator, val_generator, batch_size, optim):
    """
    Train the model
    Input:
        - net: model
        - trn_loader: dataloader for training data
        - val_loader: dataloader for validation data
        - optim: optimizer
        - scheduler: learning rate scheduler
        - num_epoch: number of epochs to train
        - collect_cycle: how many iterations to collect training statistics
        - device: device to use
        - verbose: whether to print training details
    Return:
        - best_model: the model that has the best performance on validation data
        - stats: training statistics
    """
    # Initialize:
    # -------------------------------------

    num_epoch = 40
    patience = 20
    collect_cycle = 30
    # device = 'cpu'
    verbose = True
    train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    num_itr = 0
    best_model, best_accuracy = None, 0
    num_bad_epoch = 0

    torch.manual_seed(0)
    # TODO: add in device statement!!!
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0,
                                                  end_factor=0, total_iters=20)
    loss_fn = get_loss_fn()
    if verbose:
        print('------------------------ Start Training ------------------------')
    t_start = time.time()
    for epoch in tqdm(range(num_epoch)):
        # Training:
        net.train()
        for iter, (reply, context, labels) in enumerate(tqdm(training_generator)):
            num_itr += 1
            loss = None
            ############ TODO: calculate loss, update weights ############
            # convert these to tensors when switching to colab!!!
            # context = [i.to(device) for i in context]
            # reply = [i.to(device) for i in reply]
            # labels = [i.to(device) for i in labels]
            # print("reply", reply)
            # print("context", context)
            # print("label", labels)

            optim.zero_grad()
            net._init_hidden_state()
            logits = net(reply, context)

            if(num_itr == 1):
                print("logits", logits)
            # print(logits.shape)

            flogits = logits[:, 1].flatten()
            flogits = 1 - flogits

            #flogits = flogits.float()
            # print("fLOGITS*******", flogits.shape)
            # print("lbels*******", labels.shape)
            # fake bin classification stuff for cross entropy loss

            ns_labels = 1-labels
            stacked_labels = torch.stack((ns_labels, labels))
            loss_labels = torch.transpose(stacked_labels, 0, 1).float()

            loss = calculate_loss(logits, loss_labels.float(), loss_fn)
            loss.backward()
            optim.step()

            # for p in net.parameters():
            #  print(p.grad)

            ###################### End of your code ######################

            if num_itr % collect_cycle == 0:  # Data collection cycle
                train_loss.append(loss.item())
                train_loss_ind.append(num_itr)
        if verbose:
            print('Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}'.format(
                epoch + 1,
                num_itr,
                loss.item()
            ))

        # Validation:
        accuracy, precision, recall, f1, loss = get_performance(
            net, loss_fn, val_generator, device)
        val_loss.append(loss)
        val_loss_ind.append(num_itr)
        if verbose:
            print("Validation accuracy: {:.4f}".format(accuracy))
            print("Validation precision: {:.4f}".format(precision))
            print("Validation recall: {:.4f}".format(recall))
            print("Validation f1 score: {:.4f}".format(f1))
            print("Validation loss: {:.4f}".format(loss))

        if accuracy > best_accuracy:
            best_model = copy.deepcopy(net)
            best_accuracy = accuracy
            best_acc_precision = accuracy
            best_acc_recall = recall
            best_acc_f1 = f1
            num_bad_epoch = 0
        else:
            num_bad_epoch += 1

        # early stopping
        if num_bad_epoch >= patience:
            break

        # learning rate scheduler
        scheduler.step()

    t_end = time.time()
    if verbose:
        print('Training lasted {0:.2f} minutes'.format((t_end - t_start)/60))
        print('------------------------ Training Done ------------------------')
    stats = {'train_loss': train_loss,
             'train_loss_ind': train_loss_ind,
             'val_loss': val_loss,
             'val_loss_ind': val_loss_ind,
             'accuracy': best_accuracy,
             'precision': best_acc_precision,
             'recall': best_acc_recall,
             'f1 score': best_acc_f1,
             }

    return best_model, stats


def get_performance(net, loss_fn, data_loader, device, prediction_file='prediction.txt'):
    """
    Evaluate model performance on validation set or test set.
    Input:
        - net: model
        - loss_fn: loss function
        - data_loader: data to evaluate, i.e. val or test
        - device: device to use
        - prediction_file: if not None, it's filename for the file that stores predictions
    Return:
        - accuracy: accuracy on validation set
        - loss: loss on validation set
    """

    #ffnn = FFNN(2, 50, 20)
    net.eval()
    y_true = []  # true labels
    y_pred = []  # predicted labels
    total_loss = []  # loss for each batch

    with torch.no_grad():
        for reply, context, labels in data_loader:
            loss = None  # loss for this batch
            pred = None  # predictions for this battch

            ######## TODO: calculate loss, get predictions #########
            # TODO: add back in for collab
            # context = [i.to(device) for i in context]
            # reply = [i.to(device) for i in reply]
            # labels = [i.to(device) for i in labels]

            net._init_hidden_state(len(labels))
            logits = net(reply, context)
            # print("logits ", logits)
            t_flogits = logits[:, 1].flatten()
            # t_flogits = 1 - t_flogits

            ns_labels = 1-labels
            stacked_labels = torch.stack((ns_labels, labels))
            loss_labels = torch.transpose(stacked_labels, 0, 1).float()

            loss = calculate_loss(logits, loss_labels.float(), loss_fn)

            pred = []
            #sig = nn.Sigmoid()
            #print("***", t_flogits)
            # probs = logits.cpu()
            pred = torch.round(t_flogits)

            # TODO: ADD BACK IN FOR COLLAB
            #pred = [i.to(device)for i in pred]
            #pred = torch.stack(pred)

            #pred = torch.stack(pred)
            # print("tflogits", t_flogits)
            # print("len pedictions:", len(pred))
            # print("len labels:", len(labels))

            ###################### End of your code ######################

            total_loss.append(loss.item())
            y_true.append(labels.cpu())
            y_pred.append(pred.cpu())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    accuracy = (y_true == y_pred).sum() / y_pred.shape[0]
    total_loss = sum(total_loss) / len(total_loss)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    # save predictions
    if prediction_file is not None:
        torch.save(y_pred, prediction_file)

    return accuracy, precision, recall, f1, total_loss


def search_param_basic(train_loader, dev_loader, batch_size):
    """Experiemnt on different hyper parameters."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_dim, learning_rate, weight_decay = get_hyper_parameters()
    print("hidden dimension from: {}\nlearning rate from: {}\nweight_decay from: {}".format(
        hidden_dim, learning_rate, weight_decay
    ))
    best_model, best_stats = None, None
    best_accuracy, best_lr, best_wd, best_hd = 0, 0, 0, 0
    for hd, lr, wd in tqdm(itertools.product(hidden_dim, learning_rate, weight_decay),
                           total=len(hidden_dim) * len(learning_rate) * len(weight_decay)):
        net = FFNN(hd, 8, 20).to(device)
        optim = get_optimizer(net, lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0,
                                                      end_factor=0, total_iters=40)
        model, stats = train_model(
            net, train_loader, dev_loader, batch_size, optim)
        # print accuracy
        print(f"{(hd, lr, wd)}: {stats['accuracy']}")
        # update best parameters if needed
        if stats['accuracy'] > best_accuracy:
            best_accuracy = stats['accuracy']
            best_model, best_stats = model, stats
            best_hd, best_lr, best_wd = hd, lr, wd
    print("\n\nBest hidden dimension: {}, Best learning rate: {}, best weight_decay: {}".format(
        best_hd, best_lr, best_wd))
    print("Accuracy: {:.4f}".format(best_accuracy))
    plot_loss(best_stats)
    return best_model


def plot_loss(stats):
    """Plot training loss and validation loss."""
    plt.plot(stats['train_loss_ind'],
             stats['train_loss'], label='Training loss')
    plt.plot(stats['val_loss_ind'], stats['val_loss'], label='Validation loss')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":
    train_set = 'train.csv'
    test_set = 'test.csv'
    word2vec_path = 'glove.6B.50d.txt'
    max_sent_length = 50
    max_word_length = 20

    #best_model, stats = train_model()
    # plot_loss(stats)
    #basic_model = search_param_basic()
