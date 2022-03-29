"""
@author: yee
"""
import torch
import torch.nn as nn
from src.sent_att_model import SentAttNet
from src.word_att_model import WordAttNet
from src.hierarchical_att_model import HierAttNet


class FFNN(nn.Module):
    """Basic feedforward neural network"""

    def __init__(self, hidden_dim: int):
        """
        Input:
            - hidden_dim: hidden layer dimension. Assume two hidden layers have
                the same dimension

        A few key steps of the network:
            concatenation -> linear -> relu -> linear -> relu -> linear
        """
        word_hidden_size = 50
        sent_hidden_size = 50
        batch_size = 50

        super().__init__()
        self.sig = nn.Sigmoid()
        self.fc_1 = nn.Linear(900, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, 1)
        self.reply = HierAttNet(word_hidden_size, sent_hidden_size, batch_size, training_set.num_classes,
                                opt.word2vec_path, max_sent_length, max_word_length)
        self.context = HierAttNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size, training_set.num_classes,
                                  opt.word2vec_path, max_sent_length, max_word_length)

    def forward(self, questions: List[torch.Tensor], context: List[torch.Tensor]):
        """
        Input:
            - questions: questions in each data point. Shape: [(|Q_1|, 300), (|Q_2|, 300), ...]
                where |Q_i| is the number of questions in data point i
            - context: context in each data point. Shape: [(|C_1|, 300), (|C_2|, 300), ...]
                where |C_i| is the number of context sentences in data point i
        Return:
            - output: a tensor of length |Q_1|*|C_1| + |Q_2|*|C_2| + ...
        """

        inputs = []
        # for i in range(len(questions)):
        #  for q in range(len(questions[i])):
        #    for c in range(len(context[i])):

        #      q_cj = (questions[i][q] * context[i][c])
        #      v = torch.cat((questions[i][q], context[i][c], q_cj))
        #      inputs.append(v)
        for i in range(len(questions)):
            q = questions[i]
            c = context[i]

            q = q[:, None, :]
            c = c[None, :, :]

            q_c = q * c

            q = torch.repeat_interleave(q, len(context[i]), dim=1)
            c = c.repeat(len(questions[i]), 1, 1)

            v = torch.cat((q, c, q_c), dim=2)
            v = v.flatten(end_dim=1)
            inputs.append(v)

        #v = torch.tensor(inputs, requires_grad=True)
        v = torch.cat(inputs)
        z = v.clone().detach().requires_grad_(True)
        z = F.relu(self.fc_1(z))
        z = F.relu(self.fc_2(z))
        z = self.fc_3(z)
        # print(z.shape)
        output = z.reshape((z.shape[0], ))

        #output = torch.tensor(output)
        # $output.requires_grad=True

        # rint(output.shape)
        return output
