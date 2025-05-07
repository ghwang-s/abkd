from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


class DistillKL(nn.Module):
    def __init__(self, T=1.0):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):

        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]

        return loss
