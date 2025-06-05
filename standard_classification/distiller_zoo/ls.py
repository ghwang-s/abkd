import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


class LS(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.T = opt.kd_T
        self.alpha = opt.ab_alpha
        self.beta = opt.ab_beta

    def forward(self, y_s, y_t, target, epoch):
        y_sn = normalize(y_s)
        y_tn = normalize(y_t)

        if self.alpha == 1.0 and self.beta == 0.0:
            # logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
            # logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
            temperature = self.T
            log_pred_student = F.log_softmax(y_sn / temperature, dim=1)
            pred_teacher = F.softmax(y_tn / temperature, dim=1)
            loss = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
            loss *= temperature ** 2
        else:
            p_t = F.softmax(y_tn / self.T, dim=1)
            p_s = F.softmax(y_sn / self.T, dim=1)

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
            loss = loss.mean() * (self.T ** 2)

        return loss
