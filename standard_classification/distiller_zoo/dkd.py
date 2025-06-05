import torch
import torch.nn as nn
import torch.nn.functional as F


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature, ab_alpha, ab_beta):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )

    logits_teacher_filtered = logits_teacher.masked_select(~gt_mask).view(logits_teacher.size(0), -1)
    logits_student_filtered = logits_student.masked_select(~gt_mask).view(logits_student.size(0), -1)

    # softmax
    p_t = F.softmax(logits_teacher_filtered / temperature, dim=1)
    p_s = F.softmax(logits_student_filtered / temperature, dim=1)

    if ab_alpha == 1 and ab_beta == 0:
        pred_teacher_part2 = F.softmax(logits_teacher_filtered / temperature, dim=1)
        log_pred_student_part2 = F.log_softmax(logits_student_filtered / temperature, dim=1)

        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
                * (temperature ** 2)
                / target.shape[0]
        )
    else:
        # First term: 1 / β(α + β) * log ∫ p^T(k|x_i)^(α + β) dθ
        term_1 = torch.sum((p_t ** (ab_alpha + ab_beta)), dim=1)
        term_1 = (1 / (ab_beta * (ab_alpha + ab_beta))) * term_1

        # Second term: 1 / α(α + β) * log ∫ p^S(k|x_i)^(α + β) dθ
        term_2 = torch.sum((p_s ** (ab_alpha + ab_beta)), dim=1)
        term_2 = (1 / (ab_alpha * (ab_alpha + ab_beta))) * term_2

        # Third term: - 1 / αβ * log ∫ p^T(k|x_i)^α * p^S(k|x_i)^β dk
        term_3 = torch.sum((p_t ** ab_alpha) * (p_s ** ab_beta), dim=1)
        term_3 = (-1 / (ab_alpha * ab_beta)) * term_3

        # Sum the three terms
        nckd_loss = (term_1 + term_2 + term_3).mean()

        # Scale with temperature
        nckd_loss = nckd_loss * (temperature ** 2)

    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKD(nn.Module):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, cfg):
        super().__init__()
        # self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.dkd_alpha
        self.beta = cfg.dkd_beta
        self.ab_alpha = cfg.ab_alpha
        self.ab_beta = cfg.ab_beta
        self.temperature = cfg.kd_T
        self.warmup = cfg.dkd_warmup

    def forward(self, logits_student, logits_teacher, target, epoch):
        # logits_student, _ = self.student(image)
        # with torch.no_grad():
        #     logits_teacher, _ = self.teacher(image)

        # losses
        # loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        loss_dkd = min(epoch / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            self.ab_alpha,
            self.ab_beta
        )

        return loss_dkd


class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])
