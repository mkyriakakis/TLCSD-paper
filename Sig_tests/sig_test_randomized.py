import pickle
import random
import numpy as np
import tqdm
import argparse
import copy
from sklearn.metrics import f1_score, precision_score, recall_score, auc, precision_recall_curve

def get_score(gold, predictions, metric='f1'):

    pred_targets = [int(p>0.51) for p in predictions]
    if metric == 'f1':
        return f1_score(y_true=gold, y_pred=pred_targets)
    elif metric == 'pr':
        return precision_score(y_true=gold, y_pred=pred_targets)
    elif metric == 'rc':
        return recall_score(y_true=gold, y_pred=pred_targets)
    elif metric == 'auc':
        prec, rec, thresholds = precision_recall_curve(gold,predictions)
        return auc(rec, prec)
    else:
        return None

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gold', default="", help='path to gold labels pickle')
parser.add_argument('--sysA', default="", help='path to sysA predictions pickle')
parser.add_argument('--sysB', default="", help='path to sysB predictions pickle')
parser.add_argument('--metric', default='f1', help='evaluation metric')
args = parser.parse_args()



with open(args.gold, 'rb') as f:
  gold = pickle.load(f)
with open(args.sysA, 'rb') as f:
  sysA = pickle.load(f)
with open(args.sysB, 'rb') as f:
  sysB = pickle.load(f)

sysA_metric = get_score(gold, sysA,args.metric)
print(sysA_metric)
sysB_metric = get_score(gold, sysB,args.metric)
print(sysB_metric)
orig_diff = abs(sysA_metric - sysB_metric)
print(orig_diff)

N = 10000
num_invalid = 0

for n in tqdm.tqdm(range(1, N+1)):
    sysA2 = copy.deepcopy(sysA)
    sysB2 = copy.deepcopy(sysB)
    for i in range(len(gold)):
        rval = random.random()
        if rval < 0.5:
            AD = sysA[i]
            BD = sysB[i]
            sysA2[i] = BD
            sysB2[i] = AD

    new_sysA_metric = get_score(gold, sysA2, args.metric)
    new_sysB_metric = get_score(gold, sysB2, args.metric)
    new_diff = abs(new_sysA_metric - new_sysB_metric)

    if new_diff >= orig_diff:
        num_invalid += 1

    # if n % 200 == 0 and n > 0:
    #     print('Random Iteration {}: {}'.format(n, float(num_invalid+1.0) / float(n+1.0)))

print('p-value: {}'.format(float(num_invalid+1.0) / float(N+1.0)))
