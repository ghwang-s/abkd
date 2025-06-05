from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    def __init__(self, T=1.0):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, stu_logits, tea_logits):
        L_ukd = F.kl_div(
            F.log_softmax(stu_logits / self.T, dim=1),
            F.softmax(tea_logits / self.T, dim=1),
            reduction='sum',
        ) * (self.T * self.T) / stu_logits.numel()

        return L_ukd
