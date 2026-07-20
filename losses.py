import torch
import math


class CombinedMarginLoss(torch.nn.Module):
    def __init__(self,
                 s,
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]  # Why there are -1 in labels and we must ignore those -1?
        # index_positive = labels

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits


# class ArcFace(torch.nn.Module):
#     """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
#     """
#     def __init__(self, s=9.8, margin=0.5):
#         super(ArcFace, self).__init__()
#         self.s = s
#         self.margin = margin
#         self.cos_m = math.cos(margin)
#         self.sin_m = math.sin(margin)
#         self.theta = math.cos(math.pi - margin)
#         self.sinmm = math.sin(math.pi - margin) * margin
#
#
#     def forward(self, logits: torch.Tensor, labels: torch.Tensor, margin):
#         index = torch.where(labels != -1)[0]
#         target_logit = logits[index, labels[index].view(-1)]
#
#         with torch.no_grad():
#             target_logit.arccos_()
#             logits.arccos_()
#             final_target_logit = target_logit + margin
#             logits[index, labels[index].view(-1)] = final_target_logit
#             logits.cos_()
#         return logits

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits


class NaiveFace(torch.nn.Module):
    def __init__(self, s=64.0):
        super(NaiveFace, self).__init__()
        self.s = 24

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits * self.s
        return logits


class PFace(torch.nn.Module):
    '''直接对P进行更改，暂时废弃'''

    def __init__(self, s=64.0):
        super(PFace, self).__init__()
        self.s = 24
        self.a = 1

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits * self.s

        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        exp_logits = torch.exp(logits)
        sigma = torch.sum(exp_logits, dim=1)
        P = torch.exp(target_logit) / sigma
        sigma_negative = sigma - torch.exp(target_logit)

        a = self.a
        Pm = a * P ** 3 - 2 * a * P ** 2 + P
        # Pm = (0.5 * torch.cos(P / torch.pi) + 0.5)*P
        # Pm = torch.exp(torch.log(P)/(a*P**2 - 2*a*P +1))
        self.a = a * 0.999997
        delta = 1e-9
        sigma_negative = sigma_negative.clamp(min=delta)
        Pm = Pm.clamp(delta, 0.999999)
        final_target_logit = torch.log((sigma_negative) / (1 / Pm - 1))
        logits[index, labels[index].view(-1)] = final_target_logit

        return logits


class SFace(torch.nn.Module):
    def __init__(self, s=64.0):
        super(SFace, self).__init__()
        # self.s = torch.nn.Parameter(torch.tensor(24.0))
        self.s = 24

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits * self.s
        return logits


class AdaFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.4, h=0.333, t_alpha=1.0, eps=1e-3):
        super(AdaFace, self).__init__()
        self.s = s
        self.m = m
        self.h = h
        self.t_alpha = t_alpha
        self.eps = eps

        self.register_buffer('batch_mean', torch.ones(1) * 20.0)
        self.register_buffer('batch_std', torch.ones(1) * 100.0)

    @torch.no_grad()
    def _update_batch_stats(self, safe_norms: torch.Tensor):
        mean = safe_norms.mean().detach()
        std = safe_norms.std().detach()
        self.batch_mean = mean * self.t_alpha + (1.0 - self.t_alpha) * self.batch_mean
        self.batch_std = std * self.t_alpha + (1.0 - self.t_alpha) * self.batch_std

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, norms: torch.Tensor):
        # logits: [B, C], cosine similarity before scaling
        # labels: [B], -1 for ignore
        # norms: [B] or [B, 1], L2 norm of embeddings BEFORE normalization
        index = torch.where(labels != -1)[0]
        if norms.dim() == 2 and norms.size(1) == 1:
            norms = norms.squeeze(1)
        safe_norms = torch.clamp(norms, min=0.001, max=100.0).detach()
        self._update_batch_stats(safe_norms)

        # quality-aware scaler in [-1, 1]
        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)  # 66% between -1, 1
        margin_scaler = torch.clamp(margin_scaler * self.h, -1.0, 1.0)  # 68% between -0.333 ,0.333 when h:0.333
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # clamp logits for acos stability
        logits = logits.clamp(-1.0 + self.eps, 1.0 - self.eps)

        # target positions
        target_logit = logits[index, labels[index].view(-1)]

        # 1) angular margin on target: theta -> theta + (-m * scaler)
        with torch.no_grad():
            logits.arccos_()
            target_logit = target_logit.arccos()
            final_target_theta = target_logit + (-self.m) * margin_scaler[index]
            logits[index, labels[index].view(-1)] = final_target_theta
            logits.cos_()

        # 2) additive margin on target: - (m + m*scaler)
        g_add = self.m + (self.m * margin_scaler)
        final_target_logit = logits[index, labels[index].view(-1)] - g_add[index]
        logits[index, labels[index].view(-1)] = final_target_logit

        # 3) scale
        logits = logits * self.s
        return logits


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class CurricularFace(torch.nn.Module):
    def __init__(self, in_features, out_features, m=0.5, s=64.):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        torch.nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s

        return output
