import numpy as np
import torch
import sys
import math
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, average_precision_score, roc_curve, auc


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def calculate_performance(label, pred, epoch, out_file):
    new_pred = [0 if i <=0.5 else 1 for i in pred]
    cm = confusion_matrix(label, new_pred)
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    acc = float(tp + tn) / float(tn + fn + tp + fp)
    pos_acc = float(tp) / float(tp + fp)
    neg_acc = float(tn) / float(tn + fn)
    precision=average_precision_score(label, pred)
    sensitivity = float(tp) / (tp + fn + sys.float_info.epsilon)
    specificity = float(tn) / (tn + fp + sys.float_info.epsilon)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + sys.float_info.epsilon)
    mcc = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + sys.float_info.epsilon)
    fpr, tpr, _ = roc_curve(label, pred)
    aucResults = auc(fpr, tpr)
    head_lst = ['epoch', 'tp', 'fp', 'tn', 'fn', 'acc', 'pos_acc', 'neg_acc', 'precision', 'sensitivity', 'specificity', 'f1', 'mcc', 'auc']
    Results_lst = [epoch, tp, fp, tn, fn, acc, pos_acc, neg_acc, precision, sensitivity, specificity, f1, mcc, aucResults]

    result_dict = dict(zip(head_lst, Results_lst))
    df = pd.DataFrame(data=None, columns=head_lst)
    df=df.append(result_dict, ignore_index=True)
    if epoch == 0 or isinstance(epoch, str):
        df.to_csv(out_file, mode='a', index=False)
    else:
        df.to_csv(out_file, mode='a', index=False, header=False)

    return acc, pos_acc, neg_acc
