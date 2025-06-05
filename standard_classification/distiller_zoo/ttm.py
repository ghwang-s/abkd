import torch
import torch.nn as nn
import torch.nn.functional as F


class TTM(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.alpha = opt.ab_alpha
        self.beta = opt.ab_beta
        self.gamma = opt.ab_gamma
        self.l = opt.ttm_l

    def forward(self, y_s, y_t, target, epoch):

        p_s = F.softmax(y_s, dim=1)
        p_t = torch.pow(torch.softmax(y_t, dim=1), self.l)
        norm = torch.sum(p_t, dim=1)
        p_t = p_t / norm.unsqueeze(1)

        if self.alpha == 1 and self.beta == 0:
            p_s = F.log_softmax(y_s, dim=1)
            p_t = torch.pow(torch.softmax(y_t, dim=1), self.l)
            norm = torch.sum(p_t, dim=1)
            p_t = p_t / norm.unsqueeze(1)
            KL = torch.sum(F.kl_div(p_s, p_t, reduction='none'), dim=1)
            loss = torch.mean(KL)
        else:
            # First term: 1 / β(α + β) * log ∫ p^T(k|x_i)^(α + β) dθ

            term_1 = torch.sum(p_t ** (self.alpha + self.beta), dim=1)
            term_1 = (1 / (self.beta * (self.alpha + self.beta))) * term_1
            # print(term_1)
            # Second term: 1 / α(α + β) * log ∫ p^S(k|x_i)^(α + β) dθ
            term_2 = torch.sum(p_s ** (self.alpha + self.beta), dim=1)
            term_2 = (1 / (self.alpha * (self.alpha + self.beta))) * term_2

            # Third term: - 1 / αβ * log ∫ p^T(k|x_i)^α * p^S(k|x_i)^β dk
            term_3 = torch.sum((p_t ** self.alpha) * (p_s ** self.beta), dim=1)
            term_3 = (-1 / (self.alpha * self.beta)) * term_3

            # Sum the three terms
            loss = (term_1 + term_2 + term_3)
            loss = loss.mean() 

        return loss
