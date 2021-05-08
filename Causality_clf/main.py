import time
import os
import argparse
import numpy as np
import torch
from torch import nn
import pickle
from dataloader import TextClassDataLoader
from model import RNN
from util import AverageMeter, metrics
from util import adjust_learning_rate


np.random.seed(1990)
torch.manual_seed(1990)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N', help='print frequency')
parser.add_argument('--save-freq', '-s', default=2, type=int, metavar='N', help='save frequency')
parser.add_argument('--hidden-size', default=256, type=int, metavar='N', help='rnn hidden size')
parser.add_argument('--classes', default=1, type=int, metavar='N', help='number of classes')
parser.add_argument('--layers', default=1, type=int, metavar='N', help='number of rnn layers')
parser.add_argument('--dropout', default=0.5, type=float, metavar='drp', help='dropout probability')
parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')
parser.add_argument('--w2v-tensor', default='google_w2v.pt', help='path to word2vec bin')
parser.add_argument('--w2v-voc', default='google_w2v_voc.pkl', help='path to word2vec bin')
parser.add_argument('--train-path', default="data/Semeval_train.csv", help='path to train data csv')
parser.add_argument('--dev-path', default="data/Semeval_dev.csv", help='path to dev data csv')
parser.add_argument('--rnn', default='GRU', choices=['LSTM', 'GRU'], help='rnn module type')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
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

#create trainer
print("===> creating dataloaders ...")
start = time.time()
train_loader = TextClassDataLoader(args.train_path, d_word_index, batch_size=args.batch_size,predict_flag=0,train=1)
val_loader = TextClassDataLoader(args.dev_path, d_word_index,batch_size=args.batch_size)
print('===> dataloaders creating in: {t:.2f}s'.format(t=time.time()-start))


# create model
print("===> creating rnn model ...")
vocab_size = len(d_word_index)
model = RNN(vocab_size=vocab_size, embed_size=args.embedding_size, num_output=args.classes, rnn_model=args.rnn,
            hidden_size=args.hidden_size, embedding_tensor=embed, padding_index=0, num_layers=args.layers, dropout = args.dropout,batch_first=True,cuda_flag = args.cuda)

print(model)

if args.cuda:
    model.cuda()

# optimizer and loss
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

if args.cuda:
    criterion = criterion = nn.BCEWithLogitsLoss()
else:
    criterion = criterion = nn.BCEWithLogitsLoss()


def test(val_loader, model, criterion):

    losses = AverageMeter()
    preds = []
    labels = []

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target, seq_lengths, perm_idx) in enumerate(val_loader):

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
        preds.append(output)
        labels.append(target)
        losses.update(loss.data, input.size(0))


    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    f1,pr,rc,auc = metrics(preds,labels,threshold=0.51)


    return f1,pr,rc,auc,losses.avg

batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()

# training and testing

highestScore = 0
tsid = 0
for epoch in range(1, args.epochs+1):

    adjust_learning_rate(args.lr, optimizer, epoch)
    # switch to train mode
    optimizer.zero_grad()
    steps = 0
    model.train()
    end = time.time()

    for i, (input, target, seq_lengths, perm_idx) in enumerate(train_loader):

        ts = (((epoch -1) * train_loader.n_batches) + (i+1))
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input, target, seq_lengths)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.data, input.size(0))


        # compute gradient and do SGD step
        loss.backward()
        steps+=1
        if (steps==32):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            steps = 0


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i != 0 and (i+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]  Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                  'Data {data_time.val:.2f} ({data_time.avg:.2f})  Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i+1, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses))


        if ((ts % args.save_freq == 0) and (epoch > 5)):
            start = time.time()
            f1,pr,rc,auc,loss = test(val_loader, model, criterion)
            print("Finish predictions on dev data in %.2fs" %  (time.time() - start))
            print("---\nLoss:\t%.4f\tF1-score:\t%.2f\tPrecision:\t%.2f\tRecall:\t%.2f" % (loss,f1,pr,rc))
            if f1 > highestScore:
                # save current model
                name_model = 'causality_clf_semeval'
                path_save_model = os.path.join('gen', name_model)
                torch.save({'model_state_dict': model.state_dict()}, path_save_model)
                highestScore = f1
                tsid = ts

            print("Highest F1-score: %.2f at trainning step %d" % (highestScore,tsid))
            # switch again to train mode
            model.train()

    if (steps!=0):
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
