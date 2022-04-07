"""
@author: yee
"""
import torch
import torch.nn as nn
from src.sent_att_model import SentAttNet
from src.word_att_model import WordAttNet
from src.hierarchical_att_model import HierAttNet
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul


class FFNN(nn.Module):
    """Basic feedforward neural network"""

    def __init__(self, hidden_layer_size, batch_size, max_word_length, max_sent_length):
        super().__init__()
        self.word_hidden_size = 50
        self.sent_hidden_size = hidden_layer_size
        cat_size = (self.word_hidden_size) + (self.sent_hidden_size)
        self.batch_size = batch_size
        self.num_classes = 2
        word2vec_path = 'glove.6B.50d.txt'
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        fc_size = max_word_length * 2
        self.softmax = nn.Softmax()
        self.context = HierAttNet(self.word_hidden_size, self.sent_hidden_size, self.batch_size, self.num_classes,
                                  word2vec_path, self.max_sent_length, self.max_word_length)
        self.reply = HierAttNet(self.word_hidden_size, self.sent_hidden_size, self.batch_size, self.num_classes,
                                word2vec_path, self.max_sent_length, self.max_word_length)
        self.fc = nn.Linear(2 * self.max_sent_length, self.num_classes)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.reply._init_hidden_state(batch_size)
        self.context._init_hidden_state(batch_size)

    def forward(self, reply_in, context_in):
        rep_out = self.reply(reply_in)  # .requires_grad_(True)
        # print("rep_out", rep_out.shape)
        cont_out = self.context(context_in)  # .requires_grad_(True)
        x = torch.cat((cont_out, rep_out), dim=1).requires_grad_(True)
        # print("x", x.shape)
        x = self.fc(x)
        output = F.softmax(x)
        # print("X is", x)
        return output
