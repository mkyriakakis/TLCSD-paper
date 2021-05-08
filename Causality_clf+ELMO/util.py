import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc,precision_recall_curve



''' from https://github.com/pytorch/examples/blob/master/imagenet/main.py'''
class AverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def metrics(output,target,threshold=0.51):


    pred = output >= threshold
    TP = np.sum(np.logical_and(pred == 1, target == 1))
    FP = np.sum(np.logical_and(pred == 1, target == 0))
    FN = np.sum(np.logical_and( pred == 0, target == 1))

    if TP!=0:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        fscore = 2 * (precision * recall) / (precision + recall)
    else:
        precision = 0
        recall = 0
        fscore = 0

    prec, rec, thresholds = precision_recall_curve(target,output)
    Auc = auc(rec, prec)


    return fscore,precision,recall,Auc


def adjust_learning_rate(lr, optimizer, epoch):
    """Decay LR every 20 epochs"""
    lr = lr * (0.75 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
