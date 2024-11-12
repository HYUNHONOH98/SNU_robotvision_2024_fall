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

import torch

def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1606.04797.pdf
        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]

        dc = dc.mean()

        return 1-dc
    

def calculate_pseudo_loss(raw_output,
                          threshold,
                          temp):

    prob_output = torch.sigmoid(raw_output / temp)
    pseudo_labels = (prob_output > threshold).float()

    # for unlabelled parts:
    if len(pseudo_labels.size()) == 5:
        (b, c, d, h, w) = pseudo_labels.size()
        vol = b*d*h*w*c
    elif len(pseudo_labels.size()) == 4:
        (b, d, h, w) = pseudo_labels.size()
        vol = b*d*h*w
    else:
        raise NotImplementedError

    if pseudo_labels.sum() > 0.0005*vol:
        loss_unsup = 0.5*SoftDiceLoss()(prob_output, pseudo_labels)
        loss_unsup += 0.5*F.binary_cross_entropy_with_logits(raw_output, pseudo_labels)
    else:
        loss_unsup = torch.zeros(1).cuda()

    return {'loss': loss_unsup,
            'prob': prob_output.mean()}

import math

def kld_loss(raw_output,
             mu1,
             logvar1,
             mu2=0.5,
             std2=0.125):
    '''
    Args:
        raw_output: raw digits output of predictions
        mu1: mean of posterior
        logvar1: log variance of posterior
        mu2: mean of prior
        std2: standard deviation of prior
        
    Returns:
        loss
        predicted threshold for binary pseudo labelling

    If any issues, please contact: xumoucheng28@gmail.com

    '''

    # learn the mean of posterior, separately
    mu1 = F.relu(mu1, inplace=True)

    # learn the variance of posterior
    log_sigma1 = 0.5*logvar1
    var1 = torch.exp(logvar1)
    # mean of prior
    mu2 = mu2
    sigma2 = std2

    var2 = sigma2**2
    log_sigma2 = math.log(sigma2)

    loss = log_sigma2 - log_sigma1 + 0.5 * (var1 + (mu1 - mu2)**2) / var2 - 0.5
    loss = torch.mean(torch.sum(loss, dim=-1), dim=0)

    std = torch.exp(0.5 * logvar1)
    eps = torch.randn_like(std)
    threshold = eps * std + mu1

    return loss, threshold.mean()