import torch
import torch.nn as nn
import torch.nn.functional as F


class ABKL(nn.Module):
    def __init__(self, opt):
        super(ABKL, self).__init__()
        self.T = opt.kd_T
        self.alpha = opt.ab_alpha
        self.beta = opt.ab_beta

    def forward(self, y_s, y_t):
        # Compute teacher and student probabilities
        teacher_probs = F.softmax(y_t / self.T, dim=1)
        student_probs = F.softmax(y_s / self.T, dim=1)

        # Compute student probabilities before temperature scaling
        student_probs_no_temp = F.softmax(y_s, dim=1)

        # Create inf_mask to handle infinite logits
        inf_mask = torch.isinf(y_s) | torch.isinf(y_t)

        # Special case when alpha = 0 and beta = 0
        if self.alpha == 0 and self.beta == 0:
            log_diff = torch.log(student_probs) - torch.log(teacher_probs)
            log_diff = torch.masked_fill(log_diff, inf_mask, 0)  # Handle infinities
            divergence = 0.5 * torch.sum(log_diff ** 2, dim=1)  # Use L2 divergence
        elif self.alpha == 0:
            # Case where alpha = 0
            q_beta = torch.pow(student_probs, self.beta)
            p_beta = torch.pow(teacher_probs, self.beta)
            q_beta = torch.masked_fill(q_beta, inf_mask, 0)
            p_beta = torch.masked_fill(p_beta, inf_mask, 0)
            likeli_ratio = q_beta / p_beta
            likeli_ratio = torch.masked_fill(likeli_ratio, torch.isnan(likeli_ratio), 0)
            divergence = (1 / self.beta) * torch.sum(
                q_beta * torch.log(likeli_ratio) - q_beta + p_beta,
                dim=1,
            )
        elif self.beta == 0:
            # Case where beta = 0
            p_alpha = torch.pow(teacher_probs, self.alpha)
            p_alpha = torch.masked_fill(p_alpha, inf_mask, 0)
            q_alpha = torch.pow(student_probs, self.alpha)
            q_alpha = torch.masked_fill(q_alpha, inf_mask, 0)
            divergence = (1 / self.alpha) * torch.sum(
                p_alpha * torch.log(p_alpha / q_alpha) - p_alpha + q_alpha,
                dim=1,
            )
        elif self.alpha + self.beta == 0:
            # Case where alpha + beta = 0
            p_alpha = torch.pow(teacher_probs, self.alpha)
            q_alpha = torch.pow(student_probs, self.alpha)
            p_alpha = torch.masked_fill(p_alpha, inf_mask, 0)
            q_alpha = torch.masked_fill(q_alpha, inf_mask, 0)
            divergence = torch.sum(
                (1 / self.alpha) * (torch.log(q_alpha / p_alpha) + (q_alpha / p_alpha).reciprocal() - 1),
                dim=1,
            )
        else:
            # General case
            p_alpha = torch.pow(teacher_probs, self.alpha)
            q_beta = torch.pow(student_probs, self.beta)
            p_alpha_beta = torch.pow(teacher_probs, self.alpha + self.beta)
            q_alpha_beta = torch.pow(student_probs, self.alpha + self.beta)

            # First term: - ∑ p_T^α * q_S^β
            first_term = p_alpha * q_beta
            # Second term: α / (α + β) * ∑ p_T^(α + β)
            second_term = (self.alpha / (self.alpha + self.beta)) * p_alpha_beta
            # Third term: β / (α + β) * ∑ q_S^(α + β)
            third_term = (self.beta / (self.alpha + self.beta)) * q_alpha_beta

            # Mask invalid values
            first_term = torch.masked_fill(first_term, inf_mask, 0)
            second_term = torch.masked_fill(second_term, inf_mask, 0)
            third_term = torch.masked_fill(third_term, inf_mask, 0)

            # Compute divergence
            divergence = -torch.sum(first_term - second_term - third_term, dim=1) / (self.alpha * self.beta)

        # Compute Shannon entropy for student probabilities (temperature-scaled)
        entropy_temp = -torch.sum(student_probs * torch.log(student_probs + 1e-10), dim=1)  # Avoid log(0)
        entropy_temp_mean = entropy_temp.mean()  # Mean entropy for temperature-scaled probs

        # Compute Shannon entropy for student probabilities (before temperature scaling)
        entropy_no_temp = -torch.sum(student_probs_no_temp * torch.log(student_probs_no_temp + 1e-10), dim=1)
        entropy_no_temp_mean = entropy_no_temp.mean()  # Mean entropy for unscaled probs

        # Return mean loss, mean entropy (temperature-scaled), mean entropy (unscaled)
        return divergence.mean(), entropy_temp_mean, entropy_no_temp_mean
