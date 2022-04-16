"""
Anna Ablove <aablove@umich.edu>
"""
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from src.utils import get_max_lengths, get_evaluation
from src.dataset import MyDataset
import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import train_a
from train_a import train_model
from train_a import get_optimizer
from src.final_fnn_model import FFNN
from train_a import plot_loss
import torch
from torch.utils.data import Dataset

train_set = 'train.csv'
test_set = 'test.csv'
dev_set = 'dev.csv'
word2vec_path = 'glove.6B.50d.txt'

batch_size = 64
hidden_layer_size = 128     # hidden layer size for sentence

device = 'cuda' if torch.cuda.is_available() else 'cpu'

training_params = {"batch_size": batch_size,
                   "shuffle": True,
                   "drop_last": True}
test_params = {"batch_size": batch_size,
               "shuffle": False,
               "drop_last": False}


max_word_length, max_sent_length = get_max_lengths(train_set)
#max_word_length = 50
max_sent_length = 4

print("word l, sent l", max_word_length, max_sent_length)
training_set = MyDataset(
    train_set, word2vec_path, max_sent_length, max_word_length)
training_generator = DataLoader(training_set, **training_params)

developing_set = MyDataset(
    train_set, word2vec_path, max_sent_length, max_word_length)

dev_generator = DataLoader(developing_set, **training_params)
val_generator = DataLoader(training_set, **training_params)

testing_set = MyDataset(test_set, word2vec_path,
                        max_sent_length, max_word_length)
test_generator = DataLoader(test_set, **test_params)

net = FFNN(hidden_layer_size, batch_size, max_word_length, max_sent_length)

optim = get_optimizer(net, lr=1e-3, weight_decay=1e-5)
net.to(device)

# best_model, stats = train_a.train_model(
#     net, training_generator, training_generator, batch_size, optim)

best_model, stats = train_a.search_param_basic(
    training_generator, dev_generator, batch_size, max_word_length, max_sent_length)
plot_loss(stats)
