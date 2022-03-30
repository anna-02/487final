"""
@author: yee
"""
import torch
import torch.nn as nn
from src.sent_att_model import SentAttNet
from src.word_att_model import WordAttNet
from src.hierarchical_att_model import HierAttNet
import torch.nn.functional as F


class FFNN(nn.Module):
    """Basic feedforward neural network"""

    def __init__(self, num_classes, max_sent_length, max_word_length):
        self.word_hidden_size = 50
        self.sent_hidden_size = 50
        cat_size = self.word_hidden_size + self.sent_hidden_size
        self.batch_size = 8
        self.num_classes = num_classes
        word2vec_path = 'glove.6B.50d.txt'

        super().__init__()
        self.softmax = nn.Softmax()
        self.context = HierAttNet(self.word_hidden_size, self.sent_hidden_size, self.batch_size, num_classes,
                                  word2vec_path, max_sent_length, max_word_length)
        self.reply = HierAttNet(self.word_hidden_size, self.sent_hidden_size, self.batch_size, num_classes,
                                word2vec_path, max_sent_length, max_word_length)
        self.fc = nn.Linear(cat_size, self.num_classes)

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.reply._init_hidden_state(batch_size)
        self.context._init_hidden_state(batch_size)

    def forward(self, reply_in, context_in):
        rep_out = self.reply(reply_in)
        cont_out = self.context(context_in)

        x = torch.cat((cont_out, rep_out), dim=1).requires_grad_(True)
        x = self.fc(x)
        output = F.softmax(x)
        #output, inds = torch.max(output, dim=1)
        #print("softmax first?? OUT", output.shape)
        #output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)

        #x = self.fc(x)

        #print("***** real OUT val of bev hills", output)

        #output = torch.tensor(output)
        # $output.requires_grad=True

        # rint(output.shape)
        return output
