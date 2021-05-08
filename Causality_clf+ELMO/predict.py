import time
import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from vocab import  W2VocabBuilder
from dataloader import TextClassDataLoader
from model import RNN


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency')
parser.add_argument('--hidden-size', default=128, type=int, metavar='N', help='rnn hidden size')
parser.add_argument('--layers', default=1, type=int, metavar='N', help='number of rnn layers')
parser.add_argument('--dropout', default=0, type=float, metavar='drp', help='dropout probability')
parser.add_argument('--classes', default=7, type=int, metavar='N', help='number of output classes')
parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')
parser.add_argument('--word2vec', default='data/pubmed2018_w2v_200D.bin', help='path to word2vec bin')
parser.add_argument('--test-path', default="data/Pub_type_Test_V4.csv", help='path to test data csv')
parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
parser.add_argument('--threshold', type=float, default=0.51, help='gradient clipping')
args = parser.parse_args()

print()
print("cuda flag: ", args.cuda)

print("===> creating vocabs ...")
start = time.time()
v_builder = W2VocabBuilder(args.word2vec)
d_word_index, embed = v_builder.get_word_index()
args.embedding_size = embed.size(1)

print('===> vocab creating in: {t:.3f}s'.format(t=time.time()-start))

# create trainer
print("===> creating dataloaders ...")
start = time.time()
test_loader = TextClassDataLoader(args.test_path, d_word_index,batch_size=args.batch_size,predict_flag=1)
print('===> dataloader creating in: {t:.3f}s'.format(t=time.time()-start))


# create model
print("===> creating rnn model ...")
vocab_size = len(d_word_index)
model = RNN(vocab_size=vocab_size, embed_size=args.embedding_size, num_output=args.classes, rnn_model=args.rnn,
            hidden_size=args.hidden_size, embedding_tensor=embed, padding_index=0, num_layers=args.layers, dropout = args.dropout,batch_first=True,cuda_flag = args.cuda)

print(model)


#Load model
print("===> Loading pretrained model ...")
name_model = 'hrnn_MultiLabel_V4'
model_path = os.path.join('gen', name_model)
if args.cuda:
    checkpoint = torch.load(model_path)
else:
    checkpoint = torch.load(model_path,map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])

if args.cuda:
    model.cuda()

preds = []

# switch to evaluate mode
model.eval()
start = time.time()

for i, (input, target, seq_lengths,text_indexes,text_seq_lenghts) in enumerate(test_loader):
 
    if args.cuda:
        input = input.cuda()


    # compute output
    output,target,perm_idx,_,_ = model(input, target, seq_lengths,text_indexes,text_seq_lenghts)
    s = nn.Sigmoid()
    output = output.cpu().detach()
    output = s(output).numpy()
    perm_idx = perm_idx.data.numpy().tolist()
    order_output = np.empty_like(output)

    for idx,perm in enumerate(perm_idx):
        order_output[perm,:] = output[idx,:]

    preds.append(order_output)

    if i!= 0 and i % args.print_freq == 0:
        print('Test: [{0}/{1}] elapsed time: {t:.3f}s'.format(i, len(test_loader),t=(time.time()-start)))

preds = np.concatenate(preds)
pred_data = pd.read_csv(args.test_path,sep=";")
label_names = ['clinical-trial', 'comparative-study', 'meta-analysis', 'case-reports', 'review', 'observational-study', 'rct']

try:
    del pred_data["Text"]
    del pred_data["title_abstract"]
except:
    print("Key error")

pred_data = pd.DataFrame(pred_data)
for idx,l in enumerate(label_names):
    lab = l+"_prob"
    pred_data[lab] =  preds[:,idx]

filename, file_extension = os.path.splitext(args.test_path)
pred_data.to_csv(filename+"_pred.csv",sep=";")

print("Finish predictions on test data in %.2fs" %  (time.time() - start))
