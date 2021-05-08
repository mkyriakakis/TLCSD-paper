from collections import Counter
import pandas as pd
import torch
import numpy as np
import pickle


class VocabBuilder(object):
    '''
    Read file and create word_to_index dictionary.
    '''
    def __init__(self, path_file_train,path_file_dev):
        # word count
        self.word_count = VocabBuilder.count_from_file(path_file_train,path_file_dev)
        self.word_to_index = {}

    @staticmethod
    def count_from_file(path_file_train,path_file_dev):

        df_train = pd.read_csv(path_file_train)
        if path_file_dev!=None:
            df_dev = pd.read_csv(path_file_dev)
            df = pd.concat([df_train,df_dev],sort=False)
        else:
            df = df_train
        # count
        word_count = Counter()
        for sentence in df['Sentence'].values.tolist():
            toks = sentence.split(" ")
            word_count.update([tok.lower() for tok in toks])

        print('Vocab size:{}'.format(len(word_count)))
        return word_count

    def get_word_index(self):

        """
        Returns: dict: {word_n: index_n+1, ... }

        """
        # inset padding (index=0)
        self.word_to_index = { tkn: i+1 for i, tkn in enumerate(self.word_count.keys())}
        self.word_to_index ["__PADDING__"] = 0
        words = list(self.word_to_index.keys())


        return self.word_to_index, words


class W2VocabBuilder(object) :


    def __init__(self, path_w2v):

        self.vec = None
        self.vocab = {}
        self.path_w2v = path_w2v

    def get_word_index(self, voc_path, tensor_path, padding_marker='__PADDING__', unknown_marker='__UNK__'):

        idx = 0
        with open(self.path_w2v, 'r', encoding="utf-8", newline='\n',errors='ignore') as f:
            for l in f:
                line = l.rstrip().split(' ')
                if idx == 0:
                    vocab_size = int(line[0]) + 2
                    dim = int(line[1])
                    self.vec = torch.zeros((vocab_size,dim))
                    self.vocab["__PADDING__"] = 0
                    self.vocab["__UNK__"] = 1
                    idx = 2
                else:
                    self.vocab[line[0]] = idx
                    emb = np.array(line[1:]).astype(np.float)
                    if (emb.shape[0] == dim):
                        self.vec[idx,:] = torch.tensor(np.array(line[1:]).astype(np.float))
                        idx+=1
                    else:
                        continue

        pickle.dump(self.vocab,open(voc_path,'wb'))
        torch.save(self.vec,tensor_path)

        return self.vocab, self.vec
