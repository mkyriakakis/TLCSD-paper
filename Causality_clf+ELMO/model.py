import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.modules.elmo import Elmo, batch_to_ids
import torch.nn.functional as F
from torch.autograd import Variable
from operator import itemgetter
import time


class RNN(nn.Module):

    def __init__(self, vocab_size,embed_size, num_output, rnn_model='LSTM', hidden_size=256, embedding_tensor=None,
                 padding_index=0, num_layers=1,dropout=0,elmo_cached_cnn_embeddings_voc=None, batch_first=True,cuda_flag=False):
        """

        Args:
            vocab_size: vocab size
            embed_size: dim of pretrained embeddings
            num_output: number of output (classes)
            rnn_model:  LSTM or GRU
            hidden_size: rnn hidden state dimension
            embedding_tensor: Pretrainned w2v embbedings
            padding_index: Index for padding
            hidden_size: hidden size of rnn module
            num_layers:  number of layers in rnn module
            dropout: dropout probability
            elmo_cached_cnn_embeddings_voc: a list of words to pre-compute CNN representations for elmo
            batch_first: batch first option
            cuda_flag: Flag that puts model on GPU
        """


        super(RNN, self).__init__()

        self.hidden = hidden_size
        self.dropout = dropout
        self.cuda_flag = cuda_flag

        self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index, _weight=embedding_tensor)
        self.encoder.weight.requires_grad = False
        self.dropout = self.dropout
        self.drop_en = nn.Dropout(self.dropout)


        # rnn module
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=embed_size+1024, hidden_size=hidden_size, num_layers=num_layers, dropout=0,
                                batch_first=True, bidirectional=True)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=embed_size+1024, hidden_size=hidden_size, num_layers=num_layers, dropout=0,
                                batch_first=True, bidirectional=True)
        else:
            raise LookupError(' only support LSTM and GRU')


        self.watt_weights = nn.Linear(hidden_size*2,1,bias=True)
        nn.init.xavier_uniform_(self.watt_weights.weight.data)

        self.fc = nn.Linear(hidden_size*2, num_output)

        self.elmo_weight = "elmo_original/weights.hdf5"
        self.elmo_options = "elmo_original/options.json"


        #Add vocab list to precache cnn embeddings + idx to words
        print("Cache elmo cnn embeddings")
        start = time.time()
        self.elmo = Elmo(self.elmo_options, self.elmo_weight, 1, dropout=0,vocab_to_cache=elmo_cached_cnn_embeddings_voc)
        print("Finish caching on elmo cnn embeddings in %.2fs" %  (time.time() - start))


    def forward(self,x,elmo_tensor,elmo_charIDs,seq_lengths):

        x_embed = self.encoder(x)
        sent_batch_size = elmo_charIDs.size()[0]
        elmo_embeds = []
        # s_idx = 0
        # #compute elmo representations for sentences in chunks
        # for idx in range(sent_batch_size):
        #     if ((idx%200 == 0) and (idx !=0)):
        #         elmo_embeds.append(self.elmo(elmo_charIDs[s_idx:idx,:,:],elmo_tensor[s_idx:idx,:])['elmo_representations'][0]) #
        #         s_idx = idx
        # if ((sent_batch_size - s_idx) > 0):
        elmo_embeds.append(self.elmo(elmo_charIDs,elmo_tensor)['elmo_representations'][0]) #

        elmo_embed = torch.cat((elmo_embeds),0)
        embed_concat =  torch.cat((x_embed,elmo_embed),2)

        embed_concat = self.drop_en(embed_concat)
        packed_input = pack_padded_sequence(embed_concat, seq_lengths.cpu().numpy(),batch_first=True)
        packed_output, ht = self.rnn(packed_input, None)
        out_rnn, lengths = pad_packed_sequence(packed_output, batch_first=True)
        out_rnn = self.drop_en(out_rnn)

        # apply attention layer
        weights = self.watt_weights(out_rnn)

        # create mask based on the sentence lengths
        if self.cuda_flag:
            mask = torch.ones(weights.size()).cuda()
        else:
            mask = torch.ones(weights.size())

        for i, l in enumerate(lengths):  # skip the first sentence
            if l < out_rnn.size()[1]:
                mask[i, l:] = 0


        weights = weights.masked_fill(mask == 0, -1e9)

        #apply attention and get sentence representations
        if (out_rnn.size()[1] == 1):

            attentions = F.softmax(weights,dim=1)
            weighted = torch.mul(out_rnn, attentions.expand_as(out_rnn))
            representations = weighted.sum(1)


        else:

            attentions = F.softmax(weights.squeeze(),dim=1)
            weighted = torch.mul(out_rnn, attentions.unsqueeze(-1).expand_as(out_rnn))
            representations = weighted.sum(1).squeeze()

        out = self.fc(representations).squeeze(1)

        return out
