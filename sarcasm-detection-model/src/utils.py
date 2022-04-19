"""
Anna Ablove <aablove@umich.edu>
Dan Choe <danchoe@umich.edu>
Adapted from code by: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
from sklearn import metrics
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import sys
import csv
csv.field_size_limit(sys.maxsize)


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(
            metrics.confusion_matrix(y_true, y_pred))
    return output


def matrix_mul(input, weight, bias=False):
    # print("****INSIDE MATRIX MULT INPUT IS", input)
    if bias is not False:
        bias = torch.nan_to_num(bias)
    #print("****INSIDE MATRIX MULT BIAS IS", bias)
    feature_list = []
    for feature in input:
        #print("feature:", len(feature.shape))
        if len(feature.shape) == 1:
            continue
        feature = torch.mm(feature, weight)

        # print("****INSIDE MATRIX MULT FEATURE IS", feature)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        # print("****INSIDE MATRIX MULT FEATURE IS", feature)
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)
    out = torch.cat(feature_list, 0).squeeze()
    return torch.nan_to_num(out)


def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)
    out = torch.sum(output, 0).unsqueeze(0)
    return torch.nan_to_num(out)


def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(reader):
            text = ""
            for tx in word_tokenize(line[1]):
                text += tx.lower()
                text += " "
            sent_list = sent_tokenize(text)
            sent_length_list.append(len(sent_list))

            for sent in sent_list:
                word_list = word_tokenize(sent)
                word_length_list.append(len(word_list))

        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

    return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]


if __name__ == "__main__":
    word, sent = get_max_lengths("../data/test.csv")
    print(word)
    print(sent)
