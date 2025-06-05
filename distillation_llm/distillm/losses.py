import torch
import torch.nn.functional as F


#
def ab_div(logits, teacher_logits, no_model_batch, alpha, beta):
    """
    Calculate D^{(alpha, beta)} divergence for student (logits) and teacher (teacher_logits) distributions.

    Args:
        logits: Tensor of student logits (B x S x D).
        teacher_logits: Tensor of teacher logits (B x S x D).
        no_model_batch: Dictionary containing auxiliary data (e.g., labels, mask).
        alpha: The alpha parameter in the divergence.
        beta: The beta parameter in the divergence.

    Returns:
        ab_loss: The alpha-beta divergence loss.
    """
    # Compute teacher and student probabilities
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)  # Shape: (B, S, D)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)  # Shape: (B, S, D)

    # Create inf_mask to handle infinite logits
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    # Special case when alpha = 0 and beta = 0
    if alpha == 0 and beta == 0:
        log_diff = torch.log(student_probs) - torch.log(teacher_probs)  # Shape: (B, S, D)
        log_diff = torch.masked_fill(log_diff, inf_mask, 0)  # Handle infinities
        divergence = 0.5 * torch.sum(log_diff ** 2, dim=-1)  # Shape: (B, S)
    elif alpha == 0 and beta != 0:
        # Case where alpha = 0
        q_beta = torch.pow(student_probs, beta)  # Shape: (B, S, D)
        p_beta = torch.pow(teacher_probs, beta)
        likeli_ratio = q_beta / p_beta
        likeli_ratio = torch.masked_fill(likeli_ratio, torch.isnan(likeli_ratio), 0)
        divergence = (1 / beta) * torch.sum(
            q_beta * torch.log(likeli_ratio) - q_beta + p_beta,
            dim=-1,
        )
    elif beta == 0 and alpha != 0:
        # Case where beta = 0
        p_alpha = torch.pow(teacher_probs, alpha)  # Shape: (B, S, D)
        p_alpha = torch.masked_fill(p_alpha, inf_mask, 0)
        q_alpha = torch.pow(student_probs, alpha)
        q_alpha = torch.masked_fill(q_alpha, inf_mask, 0)
        likeli_ratio = p_alpha / q_alpha
        likeli_ratio = torch.masked_fill(likeli_ratio, torch.isnan(likeli_ratio), 0)
        divergence = (1 / alpha) * torch.sum(
            p_alpha * torch.log(likeli_ratio) - p_alpha + q_alpha,
            dim=-1,
        )
    elif alpha + beta == 0:
        # Case where alpha + beta = 0
        p_alpha = torch.pow(teacher_probs, alpha)  # Shape: (B, S, D)
        q_alpha = torch.pow(student_probs, alpha)  # Shape: (B, S, D)
        p_alpha = torch.masked_fill(p_alpha, inf_mask, 0)
        q_alpha = torch.masked_fill(q_alpha, inf_mask, 0)
        divergence = torch.sum(
            (1 / alpha) * (torch.log(q_alpha / p_alpha) + (p_alpha / q_alpha) - 1),
            dim=-1
        )
    else:
        # General case
        p_alpha = torch.pow(teacher_probs, alpha)
        q_beta = torch.pow(student_probs, beta)
        divergence = p_alpha
        divergence.mul_(q_beta)
        divergence.masked_fill_(inf_mask, 0)

        tmp = torch.pow(teacher_probs, alpha + beta)
        tmp.mul_(alpha / (alpha + beta))
        tmp.masked_fill_(inf_mask, 0)
        divergence.sub_(tmp)
        del tmp

        tmp = torch.pow(student_probs, alpha + beta)
        tmp.mul_(beta / (alpha + beta))
        tmp.masked_fill_(inf_mask, 0)
        divergence.sub_(tmp)
        del tmp

        divergence = -torch.sum(divergence, dim=-1) / (alpha * beta)

    mask = (no_model_batch["label"] != -100).int()  # Shape: (B, S)

    # Apply the mask first to ignore padding positions
    masked_divergence = divergence * mask.float()  # Shape: (B, S), element-wise mask

    # Sum the divergence over the sequence length (S), resulting in shape (B,)
    x = torch.sum(masked_divergence, dim=-1)  # Sum over the sequence dimension (S)

    # Compute the ab_loss by summing the masked loss and normalizing by the number of valid positions
    ab_loss = torch.sum(x) / torch.sum(mask.float())  # Normalize by the total number of valid tokens

    return ab_loss


def bdkd(logits, teacher_logits, no_model_batch):
    def entropy(logits):
        """计算 softmax 概率分布的熵"""
        probs = F.softmax(logits, dim=-1)  # 转换为概率
        log_probs = torch.log(probs + 1e-9)  # 计算 log 概率，防止数值问题
        return -torch.sum(probs * log_probs, dim=-1)  # 计算熵

    # 计算学生和教师 logits 的熵
    entropy_student = entropy(logits)  # (B, S)
    entropy_teacher = entropy(teacher_logits)  # (B, S)

    # 生成权重矩阵
    weight_student = torch.where(entropy_student > entropy_teacher, 3.0, 1.0)  # 学生熵大则设为2，否则设为1
    weight_teacher = torch.where(entropy_teacher > entropy_student, 3.0, 1.0)  # 教师熵大则设为2，否则设为1

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss1 = -torch.sum(x * mask.view(-1) * weight_teacher.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss2 = -torch.sum(x * mask.view(-1) * weight_student.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    return distil_loss1 + distil_loss2


def forward_kl(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def reverse_kl(logits, teacher_logits, no_model_batch):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def get_ratio(teacher_logits, logits, mu=0.5):
    # [B, L, V]
    teacher_logits = torch.masked_fill(teacher_logits, torch.isinf(teacher_logits), 0).to(torch.float32)
    logits = torch.masked_fill(logits, torch.isinf(logits), 0).to(torch.float32)

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32).detach()

    re_teacher_probs, idx = teacher_probs.sort(dim=-1, descending=True)
    re_student_probs = student_probs.gather(dim=-1, index=idx)

    errors = torch.abs(re_teacher_probs - re_student_probs)

    cum_sum = torch.cumsum(re_teacher_probs, dim=-1)  # B,L,V
    mask = cum_sum > mu
    mask[:, :, 0] = False  # 第一个概率一定要置False，对应第一个概率>0.5时mask全True

    s1 = torch.masked_fill(errors, mask, 0.0).sum(dim=-1)
    s2 = torch.masked_fill(errors, ~mask, 0.0).sum(dim=-1)

    return s1 / (s1 + s2), s2 / (s1 + s2)


def get_kl(teacher_logits, logits, inf_mask, mask, ratio=None):
    # ratio: [B,L]
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_prod_probs = torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    teacher_x = torch.sum(teacher_prod_probs, dim=-1).view(-1)

    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)  # [B,L]->[BL]

    if ratio == None:
        distil_loss = torch.sum((teacher_x - x) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    else:
        distil_loss = torch.sum((teacher_x - x) * ratio.view(-1) * mask.view(-1), dim=0) / torch.sum(mask.view(-1),
                                                                                                     dim=0)
    return distil_loss


def AKL(teacher_logits, logits, no_model_batch):
    inf_mask = torch.isinf(logits)  # [batch, seq, vocab]
    mask = (no_model_batch["label"] != -100).int()  # [batch, seq]

    h_ratio, l_ratio = get_ratio(teacher_logits, logits)
    distil_loss = get_kl(teacher_logits, logits, inf_mask, mask, h_ratio) + get_kl(logits, teacher_logits, inf_mask,
                                                                                   mask, l_ratio)
    return distil_loss


def symmetric_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    for_kl = forward_kl(logits, teacher_logits, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    distil_loss = (1 - lam) * for_kl + lam * rev_kl
    return distil_loss


def js_distance(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1 - lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss += (1 - lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    return distil_loss


def wsd(logits, teacher_logits, no_model_batch):
    for_kl = forward_kl(logits, teacher_logits, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    distil_loss = (1 - lam) * for_kl + lam * rev_kl

    return distil_loss


def tv_distance(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1 - lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    r_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1)
    return 0.5 * distil_loss + 0.5 * r_loss


def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1 - lam) * teacher_probs + lam * student_probs

    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def f_divergence(q_logits, p_logits, alpha, iw_clip=1e3):
    # assert isinstance(alpha, float)
    inf_mask = torch.isinf(q_logits) | torch.isinf(p_logits)
    q_logits = torch.masked_fill(q_logits, inf_mask, 0)
    p_logits = torch.masked_fill(p_logits, inf_mask, 0)
    q_prob = torch.nn.functional.softmax(q_logits, dim=-1).detach()
    p_prob = torch.nn.functional.softmax(p_logits, dim=-1).detach()
    q_log_prob = torch.nn.functional.log_softmax(q_logits, dim=-1)  # gradient is only backpropagated here

    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
        importance_ratio = importance_ratio.clamp(0, iw_clip)
        f = -importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio.log() - 1.0
    elif abs(alpha - 1.0) < 1e-3:
        f = importance_ratio * importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = torch.pow(importance_ratio, alpha)
        iw_alpha = iw_alpha.clamp(0, iw_clip)
        f = iw_alpha / alpha / (alpha - 1.0)
        f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha / alpha + f_base

    loss = torch.sum(q_prob * (f - f_base), dim=-1)
    grad_loss = -torch.sum(q_prob * rho_f * q_log_prob, dim=-1)
    return loss, grad_loss


def alphanet(logits, teacher_logits, no_model_batch, alpha, beta):
    loss1 = ab_div(logits, teacher_logits, no_model_batch, alpha, 1 - alpha)
    loss2 = ab_div(logits, teacher_logits, no_model_batch, beta, 1 - beta)
    if loss1 > loss2:
        return loss1
    return loss2
