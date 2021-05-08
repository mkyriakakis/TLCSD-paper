import torch
import pandas as pd
import numpy as np
import math
import random
from allennlp.modules.elmo import batch_to_ids


class TextClassDataLoader(object):

    def __init__(self, path_file, word_to_index, word_to_index_elmo, batch_size=32 ,predict_flag=0, train=0):
        """

        Args:
            path_file:
            word_to_index:
            batch_size:
        """

        self.batch_size = batch_size
        self.word_to_index = word_to_index
        self.word_to_index_elmo = word_to_index_elmo
        self.predict_flag = predict_flag
        self.train = train
        print("Train flag: ",self.train)
        print("Predict flag: ",self.predict_flag)
        # read file
        df = pd.read_csv(path_file)
        df['Text_id'] = df['Sentence'].apply(self.generate_indexifyer())
        df['Text_id_elmo'] = df['Sentence'].apply(self.generate_indexifyer_elmo())
        if self.predict_flag:
            data = df[['Sentence','Text_id','Text_id_elmo']]
        else:
            data = df[['Sentence','Text_id','Text_id_elmo','Annotated_Causal']]
        self.samples = data.values.tolist()

        # for batch
        self.n_samples = len(self.samples)
        self.n_batches = math.ceil(self.n_samples / self.batch_size)
        self.max_length = self._get_max_length()
        self.index = 0
        self.batch_index = 0
        self.indices = np.arange(self.n_samples)
        if self.train:
            self._shuffle_indices()

        self.report()

    def _shuffle_indices(self):
        self.indices = np.random.permutation(self.n_samples)
        self.index = 0
        self.batch_index = 0

    def _get_max_length(self):
        length = 0
        for sample in self.samples:
            length = max(length, len(sample[0]))
        return length

    def generate_indexifyer(self):

        def indexify(sentence):
            indices = []
            toks = sentence.split()
            for tok in toks:
                if tok.lower() in self.word_to_index:
                    indices.append(self.word_to_index[tok.lower()])
                else:
                    indices.append(self.word_to_index['__UNK__'])

            return indices

        return indexify

    def generate_indexifyer_elmo(self):

        def indexify(sentence):
            indices = []
            toks = sentence.split()
            for tok in toks:
                indices.append(self.word_to_index_elmo[tok.lower()])
            return indices

        return indexify

    def _create_batch(self):

        batch = []
        n = 0
        while ((n < self.batch_size) and (self.index < len(self.samples))):
            _index = self.indices[self.index]
            batch.append(self.samples[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        #Fix for the extreme case that last batch has size == 1. Append the last sample to the
        #previous batch
        if (self.index+1 == len(self.samples)):
            _index = self.indices[self.index]
            batch.append(self.samples[_index])
            self.index += 1
            self.batch_index += 1

        text = []
        sentences = []
        elmo = []
        labels = []

        for bat in batch:
            if not self.predict_flag:
                labels.append(bat[3])
            text.append(bat[0])
            sentences.append(bat[1])
            elmo.append(bat[2])


        seq_lengths = torch.LongTensor(list(map(len, sentences)))

        #padding
        seq_tensor = torch.zeros((len(sentences), seq_lengths.max())).long()
        elmo_tensor = torch.zeros((len(elmo), seq_lengths.max())).long()
        for idx, (seq,elmo, seqlen) in enumerate(zip(sentences,elmo, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            elmo_tensor[idx, :seqlen] = torch.LongTensor(elmo)

        #sort in decreasing order
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        elmo_tensor = elmo_tensor[perm_idx]

        sort_idx = perm_idx.data.numpy().tolist()
        text_sorted = []
        for idx in sort_idx:
            text_sorted.append(text[idx])

        if not self.predict_flag:
            labels = torch.FloatTensor(labels)
            labels = labels[perm_idx]

        else:
            labels = None

        elmo_charIDs = batch_to_ids(text_sorted)

        return seq_tensor, elmo_tensor, elmo_charIDs, labels, seq_lengths, perm_idx


    def __len__(self):
        return self.n_batches

    def __iter__(self):

        if self.train:
            self._shuffle_indices()
        else:
            self.index = 0
            self.batch_index = 0

        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()

    def show_samples(self, n=10):
        for sample in self.samples[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.samples)))
        print('max len: {}'.format(self.max_length))
        print('# vocab: {}'.format(len(self.word_to_index)))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))
