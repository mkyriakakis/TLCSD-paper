import time
import os
import argparse
import numpy as np
import torch
from torch import nn
import pickle
from vocab import  VocabBuilder
from dataloader import TextClassDataLoader
from model import RNN
from util import AverageMeter, metrics


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency')
parser.add_argument('--hidden-size', default=256, type=int, metavar='N', help='rnn hidden size')
parser.add_argument('--layers', default=1, type=int, metavar='N', help='number of rnn layers')
parser.add_argument('--dropout', default=0, type=float, metavar='drp', help='dropout probability')
parser.add_argument('--classes', default=1, type=int, metavar='N', help='number of output classes')
parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')
parser.add_argument('--embed-tensor', default='data/', help='path to word2vec bin')
parser.add_argument('--embed-voc', default='data/', help='path to word2vec bin')
parser.add_argument('--test-path', default="data/", help='path to test data csv')
parser.add_argument('--model', default='No', help='pretrained model name')
parser.add_argument('--rnn', default='GRU', choices=['LSTM', 'GRU'], help='rnn module type')
parser.add_argument('--threshold', type=float, default=0.51, help='threshold value')
args = parser.parse_args()

print()
print("cuda flag: ", args.cuda)

# Load pretrained embeddings
start = time.time()
print("===> Loading pretrained embeddings ...")
end = time.time()
d_word_index = pickle.load(open(args.embed_voc,"rb"))
embed = torch.load(args.embed_tensor)
args.embedding_size = embed.size(1)
print('===> Loaded pretrained embeddings in: {t:.2f}s'.format(t=time.time()-start))

# create ELMo cached vocab
start = time.time()
print("===> creating vocabs ...")
v_builder_elmo = VocabBuilder(path_file_train=args.test_path,path_file_dev=None)
d_word_index_elmo, word_list = v_builder_elmo.get_word_index()
d_index_to_word_elmo = { i: tkn for tkn, i in d_word_index_elmo.items()}
elmo_cached_cnn_embeddings = []
for idx in range(len(d_word_index_elmo)):
    elmo_cached_cnn_embeddings.append(d_index_to_word_elmo[idx])

print('===> vocab creating in: {t:.2f}s'.format(t=time.time()-start))


# create dataloader
print("===> creating dataloaders ...")
start = time.time()
test_loader = TextClassDataLoader(args.test_path, d_word_index,d_word_index_elmo,batch_size=args.batch_size)
print('===> dataloader creating in: {t:.3f}s'.format(t=time.time()-start))


# create model
print("===> creating rnn model ...")
vocab_size = len(d_word_index)
model = RNN(vocab_size=vocab_size, embed_size=args.embedding_size, num_output=args.classes, rnn_model=args.rnn,
            hidden_size=args.hidden_size, embedding_tensor=embed, padding_index=0, num_layers=args.layers, dropout = args.dropout,
            elmo_cached_cnn_embeddings_voc= elmo_cached_cnn_embeddings,batch_first=True,cuda_flag = args.cuda)

print(model)
criterion = nn.BCEWithLogitsLoss()

#Load model
print("===> Loading pretrained model {} ...".format(args.model))
name_model = os.path.basename(args.model)
model_path = os.path.join('gen', name_model)
if args.cuda:
    checkpoint = torch.load(model_path)
else:
    checkpoint = torch.load(model_path,map_location="cpu")
checkpoint['model_state_dict']["elmo._elmo_lstm._word_embedding.weight"] = model.elmo._elmo_lstm._word_embedding.weight
model.load_state_dict(checkpoint['model_state_dict'])

if args.cuda:
    model.cuda()

preds = []
labels = []
losses = AverageMeter()

# switch to evaluate mode
model.eval()
start = time.time()

for i, (tensor, elmo_tensor, elmo_charIDs, target, seq_lengths, perm_idx) in enumerate(test_loader):

    if args.cuda:
        tensor = tensor.cuda()
        elmo_tensor = elmo_tensor.cuda()
        target = target.cuda()

    # compute output
    output = model(tensor,elmo_tensor,elmo_charIDs,seq_lengths)
    loss = criterion(output, target)
    s = nn.Sigmoid()
    output = output.cpu().detach()
    output = s(output).numpy()
    target = target.cpu().numpy()
    preds.append(output)
    labels.append(target)
    losses.update(loss.data, tensor.size(0))



    if i!= 0 and i % args.print_freq == 0:
        print('Test: [{0}/{1}]  Elapsed Time {time}  '
              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
               i, len(test_loader), time=time.time()-start, loss=losses))

preds = np.concatenate(preds)
labels = np.concatenate(labels)
f1,pr,rc,auc = metrics(preds,labels,threshold=0.51)

print("Finish predictions on test data in %.2fs" %  (time.time() - start))
print("---\nF1-score:\t%.4f\tPrecision:\t%.4f\tRecall:\t%.4f\tAUC:\t%.4f\tAvg_loss:\t%.4f" % (f1,pr,rc,auc,losses.avg))
