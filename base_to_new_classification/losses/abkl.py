import torch
import torch.nn as nn
import torch.nn.functional as F


class ABKL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.T = cfg.TRAINER.PROMPTKD.TEMPERATURE
        self.alpha = cfg.ab_alpha
        self.beta = cfg.ab_beta
        # self.gamma = cfg.ab_gamma

    def forward(self, y_s, y_t):
        p_t = F.softmax(y_t / self.T, dim=1)
        p_s = F.softmax(y_s / self.T, dim=1)

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

        return loss.mean()
