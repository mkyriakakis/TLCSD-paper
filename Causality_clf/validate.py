import time
import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
import pickle
from vocab import  W2VocabBuilder
from dataloader import TextClassDataLoader
from model import RNN
from util import AverageMeter, metrics


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency')
parser.add_argument('--hidden-size', default=256, type=int, metavar='N', help='rnn hidden size')
parser.add_argument('--layers', default=1, type=int, metavar='N', help='number of rnn layers')
parser.add_argument('--dropout', default=0, type=float, metavar='drp', help='dropout probability')
parser.add_argument('--classes', default=1, type=int, metavar='N', help='number of output classes')
parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')
parser.add_argument('--w2v_tensor', default='google_w2v.pt', help='path to word2vec bin')
parser.add_argument('--w2v_voc', default='google_w2v_voc.pkl', help='path to word2vec bin')
parser.add_argument('--test-path', default="data/Causal_TimeBank_test.csv", help='path to test data csv')
parser.add_argument('--rnn', default='GRU', choices=['LSTM', 'GRU'], help='rnn module type')
parser.add_argument('--threshold', type=float, default=0.51, help='threshold value')
args = parser.parse_args()

print()
print("cuda flag: ", args.cuda)

# create vocabs
start = time.time()
print("===> Loading pretrained embeddings ...")
end = time.time()
d_word_index = pickle.load(open(args.w2v_voc,"rb"))
embed = torch.load(args.w2v_tensor)
args.embedding_size = embed.size(1)

print('===> Loaded pretrained embeddings in: {t:.2f}s'.format(t=time.time()-start))


# create trainer
print("===> creating dataloaders ...")
start = time.time()
test_loader = TextClassDataLoader(args.test_path, d_word_index,batch_size=args.batch_size)
print('===> dataloader creating in: {t:.3f}s'.format(t=time.time()-start))


# create model
print("===> creating rnn model ...")
vocab_size = len(d_word_index)
model = RNN(vocab_size=vocab_size, embed_size=args.embedding_size, num_output=args.classes, rnn_model=args.rnn,
            hidden_size=args.hidden_size, embedding_tensor=embed, padding_index=0, num_layers=args.layers, dropout = args.dropout,batch_first=True,cuda_flag = args.cuda)

criterion = nn.BCEWithLogitsLoss()
print(model)

#Load model
print("===> Loading pretrained model ...")
name_model = 'causality_clf_semeval'
model_path = os.path.join('gen', name_model)
if args.cuda:
    checkpoint = torch.load(model_path)
else:
    checkpoint = torch.load(model_path,map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])

if args.cuda:
    model.cuda()

preds = []
labels = []
losses = AverageMeter()

# switch to evaluate mode
model.eval()
start = time.time()

for i, (input, target, seq_lengths, perm_idx) in enumerate(test_loader):

    if args.cuda:
        input = input.cuda()
        target = target.cuda()

    # compute output
    output = model(input, target, seq_lengths)
    loss = criterion(output, target)
    s = nn.Sigmoid()
    output = output.cpu().detach()
    output = s(output).numpy()
    target = target.cpu().numpy()
    order_output = np.empty_like(output)
    order_target = np.empty_like(target)
    for idx,perm in enumerate(perm_idx):
        order_output[perm] = output[idx]
        order_target[perm] = target[idx]
    preds.append(order_output)
    labels.append(order_target)
    losses.update(loss.data, input.size(0))



    if i!= 0 and i % args.print_freq == 0:
        print('Test: [{0}/{1}]  Elapsed Time {time}  '
              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
               i, len(test_loader), time=time.time()-start, loss=losses))

preds = np.concatenate(preds)
labels = np.concatenate(labels)
f1,pr,rc,auc = metrics(preds,labels,threshold=0.51)

print("Finish predictions on test data in %.2fs" %  (time.time() - start))
print("---\nF1-score:\t%.2f\tPrecision:\t%.2f\tRecall:\t%.2f\tAUC:\t%.2f\tAvg_loss:\t%.4f" % (f1,pr,rc,auc,losses.avg))
