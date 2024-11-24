import numpy as np
from itertools import filterfalse as  ifilterfalse
from .save_pseudo import save_pseudo_labels
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
#     """
#     net_output must be (b, c, x, y(, z)))
#     gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
#     if mask is provided it must have shape (b, 1, x, y(, z)))
#     :param net_output:
#     :param gt:
#     :param axes:
#     :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
#     :param square: if True then fp, tp and fn will be squared before summation
#     :return:
#     """
#     if axes is None:
#         axes = tuple(range(2, len(net_output.size())))

#     shp_x = net_output.shape
#     shp_y = gt.shape

#     with torch.no_grad():
#         if len(shp_x) != len(shp_y):
#             gt = gt.view((shp_y[0], 1, *shp_y[1:]))

#         if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
#             # if this is the case then gt is probably already a one hot encoding
#             y_onehot = gt
#         else:
#             gt = gt.long()
#             y_onehot = torch.zeros(shp_x)
#             if net_output.device.type == "cuda":
#                 y_onehot = y_onehot.cuda(net_output.device.index)
#             y_onehot.scatter_(1, gt, 1)

#     tp = net_output * y_onehot
#     fp = net_output * (1 - y_onehot)
#     fn = (1 - net_output) * y_onehot

#     if mask is not None:
#         tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
#         fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
#         fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

#     if square:
#         tp = tp ** 2
#         fp = fp ** 2
#         fn = fn ** 2

#     tp = sum_tensor(tp, axes, keepdim=False)
#     fp = sum_tensor(fp, axes, keepdim=False)
#     fn = sum_tensor(fn, axes, keepdim=False)

#     return tp, fp, fn

def get_tp_fp_fn(logit, label, axes, mask):
    """
    Computes true positives, false positives, and false negatives.

    Parameters:
    - logit: torch.Tensor of shape [batch_size, num_classes, H, W]
    - label: torch.Tensor of shape [batch_size, H, W], containing labels from 0 to num_classes-1 and possibly an ignore label (e.g., 255)
    - mask: torch.Tensor of shape [batch_size, H, W], where positions with ignore label are 0, others are 1
    - axes: tuple of axes over which to sum (e.g., (0, 2, 3))

    Returns:
    - tp: torch.Tensor of shape [num_classes], true positives per class
    - fp: torch.Tensor of shape [num_classes], false positives per class
    - fn: torch.Tensor of shape [num_classes], false negatives per class
    """
    num_classes = logit.shape[1]
    num_classes += 1  # Add an extra class for ignored labels

    # Get predicted classes
    preds = torch.argmax(logit, dim=1)  # Shape: [batch_size, H, W]

    # Handle ignored labels by assigning them to the extra class
    label = label.clone()
    label[mask == 0] = num_classes - 1  # Assign ignore label to extra class

    # Create one-hot encodings
    preds_one_hot = F.one_hot(preds, num_classes=num_classes).float()  # Shape: [batch_size, H, W, num_classes]
    labels_one_hot = F.one_hot(label, num_classes=num_classes).float()  # Shape: [batch_size, H, W, num_classes]

    # Ignore the extra class in calculations
    preds_one_hot = preds_one_hot[:, :, :, :num_classes - 1]
    labels_one_hot = labels_one_hot[:, :, :, :num_classes - 1]

    # Permute to match shape [batch_size, num_classes-1, H, W]
    preds_one_hot = preds_one_hot.permute(0, 3, 1, 2)
    labels_one_hot = labels_one_hot.permute(0, 3, 1, 2)

    # Expand mask to match dimensions
    mask = mask.unsqueeze(1)  # Shape: [batch_size, 1, H, W]

    # Compute TP, FP, FN
    tp = (preds_one_hot * labels_one_hot * mask).sum(dim=axes)
    fp = (preds_one_hot * (1 - labels_one_hot) * mask).sum(dim=axes)
    fn = ((1 - preds_one_hot) * labels_one_hot * mask).sum(dim=axes)

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

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]

        dc = dc.mean()

        return 1-dc

import os
def calculate_pseudo_loss(teacher_output,
                          student_output,
                          threshold,
                          save_info: dict,
                          args
                          ):
    epoch, names = save_info["epoch"], save_info["name"]

    save_dir = os.path.join("/home/hyunho/sfda/pseudo_label", args.exp_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, str(epoch))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    stu_prob_output = torch.softmax(student_output, dim=1)
    tut_prob_output = torch.softmax(teacher_output, dim=1)
    max_prob, max_class = tut_prob_output.max(dim=1)
    threshold_map = threshold[max_class]
    pseudo_labels = max_class.clone()
    pseudo_labels[max_prob < threshold_map] = 255
    # TODO pseudo label 시각화하기

    background_mask = (pseudo_labels != 255).long()
    pseudo_labels = pseudo_labels.long()

    save_pseudo_labels(names, pseudo_labels, save_dir)

    # for unlabelled parts:
    if len(pseudo_labels.size()) == 3:
        (b, h, w) = pseudo_labels.size()
        vol = b*h*w
    else:
        raise NotImplementedError

    if pseudo_labels.sum() > 0.0005*vol:
        loss_unsup = 0.5*SoftDiceLoss()(stu_prob_output, pseudo_labels, background_mask)
        # print(loss_unsup)
        loss_unsup += 0.5*nn.CrossEntropyLoss(ignore_index=255)(student_output, pseudo_labels)
        # print(loss_unsup)
    else:
        loss_unsup = torch.zeros(1).cuda()

    return {'loss': loss_unsup,
            'prob': stu_prob_output.mean()}


import math

def kld_loss(mu1,
             logvar1,
             mu2=[0.65, 0.51, 0.8, 0.5, 0.51, 0.66, 0.68, 0.73, 0.8, 0.69, 0.8, 0.73, 0.5, 0.8, 0.56, 0.5, 0.5, 0.5, 0.8],
             std2=0.125):
    '''
    Args:
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
    # mean of prior # TODO tensor 로 바꾸기
    mu2 = torch.tensor(mu2).to(device)
    sigma2 = torch.full((mu1.shape[1],), std2, device=mu1.device).to(device)
    mu2 = mu2.view(1, 19, 1)
    sigma2 = sigma2.view(1, 19, 1)


    var2 = sigma2**2
    log_sigma2 = torch.log(sigma2)

    loss = log_sigma2 - log_sigma1 + 0.5 * (var1 + (mu1 - mu2)**2) / var2 - 0.5
    # loss = torch.mean(torch.sum(loss, dim=-1), dim=0) # ORIGINAL
    loss = torch.mean(torch.mean(torch.sum(loss, dim=-1), dim=-1), dim=0)

    std = torch.exp(0.5 * logvar1)
    eps = torch.randn_like(std)
    threshold = eps * std + mu1

    # return loss, threshold.mean() # ORIGINAL
    return {
        "loss":loss, 
        "threshold":threshold.mean(dim=(0,2))
        } # CHANGE multi-channel threshold