# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F

class GHM_Loss(nn.Module):
    def __init__(self, bins=10, alpha=0.5):
        '''
        bins: split to n bins
        alpha: hyper-parameter
        '''
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        beta = beta.type_as(x)

        return self._custom_loss(x, target, beta[bin_idx])


class GHMC_Loss(GHM_Loss):
    '''
        GHM_Loss for classification
    '''
    def __init__(self, bins, alpha):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target


class GHMR_Loss(GHM_Loss):
    '''
        GHM_Loss for regression
    '''

    def __init__(self, bins, alpha, mu):
        super(GHMR_Loss, self).__init__(bins, alpha)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        N = x.size(0) * x.size(1)
        return (loss * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)

def FocalLoss(y_pred, y_true, alpha=0.25, gamma=2):
    if y_pred.shape != y_true.shape:
        y_true = y_true.flatten()
    y_true = y_true.long()
    y_pred = y_pred.float()
    y_true = y_true.float()
    y_true = y_true.unsqueeze(1)
    y_pred = y_pred.unsqueeze(1)
    y_true = torch.cat((1-y_true, y_true), dim=1)
    y_pred = torch.cat((1-y_pred, y_pred), dim=1)
    y_pred = y_pred.clamp(1e-5, 1.0)
    loss = -alpha * y_true * torch.pow((1 - y_pred), gamma) * torch.log(y_pred)
    return torch.mean(torch.sum(loss, dim=1))

def FocalLossWithLogits(y_pred, y_true, alpha=0.25, gamma=2.0):
    y_pred = torch.sigmoid(y_pred)
    return FocalLoss(y_pred, y_true, alpha, gamma)

def myCrossEntropyLoss(y_pred, y_true):
    if y_pred.shape != y_true.shape:
        y_true = y_true.flatten()
    return nn.CrossEntropyLoss()(y_pred, y_true)