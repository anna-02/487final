"""
Anna Ablove <aablove@umich.edu>
Dan Choe <danchoe@umich.edu>
Adapted from code by: Viet Nguyen <nhviet1009@gmail.com>
"""
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from csv import reader
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


class MyDataset(Dataset):

    def __init__(self, data_path, dict_path, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()
        self.contexts = []
        self.replies = []
        self.labels = []

        replies, contexts, labels = [], [], []
        with open(data_path) as read_obj:
            reader = csv.reader(read_obj, quotechar='"')
            next(reader)  # skipping first line
            for idx, line in enumerate(reader):
                # build replies
                reply = ""
                for tx in word_tokenize(line[1]):
                    reply += tx.lower()
                    reply += " "

                # build context
                context = ""
                for tx in word_tokenize(line[2]):
                    context += tx.lower()
                    context += " "

                label = int(line[0])
                self.contexts.append(context)
                self.replies.append(reply)
                self.labels.append(label)

        # self.contexts = contexts
        # self.replies = replies
        # self.labels = labels
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.contexts[index]
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
            in
            sent_tokenize(text=text)]

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(
                    self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
            :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        context_ = document_encode.astype(np.int64)

        text = self.replies[index]
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
            in
            sent_tokenize(text=text)]

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(
                    self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
            :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        reply_ = document_encode.astype(np.int64)
        return reply_, context_, label


if __name__ == '__main__':
    test = MyDataset(data_path="../data/test.csv",
                     dict_path="../data/glove.6B.50d.txt")
    print(test.__getitem__(index=1)[0].shape)
