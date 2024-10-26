import numpy as np
from itertools import filterfalse as  ifilterfalse

def isnan(x):
    return x != x

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(x) for x in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
class HLoss(nn.Module):
    """
    Weighted Entropy Loss with ignore index support
    """
    def __init__(self, mode: Literal["sum","mean"] = "mean"):
        super(HLoss, self).__init__()
        self.mode = mode

    def forward(self, x):
        # Calculate entropy loss
        B,C,H,W = x.shape
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)  # Shape: (B, H, W)
        # Apply weighting function and mean reduction
        if self.mode == "mean":
            b = self.weight_function(b).mean()
        else:
            b = self.weight_function(b).sum()
        return b
    
    def weight_function(self, x, eta=2.0):
        return (x**2 + 1e-6)**eta
