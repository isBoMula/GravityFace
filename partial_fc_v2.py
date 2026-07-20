import math

import torch
import torch.nn.functional as F
from losses import AdaFace, ArcFace, CosFace, SFace


class DualViewGravitationalLoss(torch.nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        fp16: bool = False,
        sample_gamma: float = 0.5,
        class_gamma: float = 0.5,
        sample_alpha: float = 0.2,
        class_alpha: float = 0.2,
        min_weight: float = 0.7,
        max_weight: float = 1.2,
        class_mass_momentum: float = 0.7,
        eps: float = 1e-6,
    ):
        super(DualViewGravitationalLoss, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.fp16 = fp16
        self.sample_gamma = sample_gamma
        self.class_gamma = class_gamma
        self.sample_alpha = sample_alpha
        self.class_alpha = class_alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.class_mass_momentum = class_mass_momentum
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))

        # Keep the existing usage pattern and apply asymmetric mass weighting
        # after the margin transform.
        # self.logit = AdaFace()
        self.logit = ArcFace()
        # self.logit = CosFace()
        # self.logit = SFace()

        self.register_buffer("class_mass", torch.ones(1, num_classes))

    def _balanced_mass_weight(
        self,
        mass: torch.Tensor,
        normalize_dim: int,
        gamma: float,
        alpha: float,
    ) -> torch.Tensor:
        safe_mass = mass.clamp_min(self.eps)
        relative_mass = safe_mass / (safe_mass.mean(dim=normalize_dim, keepdim=True) + self.eps)
        compressed_mass = torch.pow(relative_mass, gamma)
        recentered_mass = compressed_mass / (
            compressed_mass.mean(dim=normalize_dim, keepdim=True) + self.eps
        )
        balanced_mass = 1.0 + alpha * (recentered_mass - 1.0)
        return balanced_mass.clamp(self.min_weight, self.max_weight)

    def _run_logit(
        self,
        cosine: torch.Tensor,
        labels: torch.Tensor,
        norms: torch.Tensor,
    ) -> torch.Tensor:
        logits = cosine.clone()
        if isinstance(self.logit, AdaFace):
            return self.logit(logits, labels, norms)
        if isinstance(self.logit, (ArcFace, CosFace, SFace)):
            return self.logit(logits, labels)
        raise TypeError(
            "DualViewGravitationalLoss currently supports SFace, ArcFace, CosFace, and AdaFace."
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.any().item():
            return values[mask].mean()
        return values.sum() * 0.0

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        raw_m_i = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        m_i = raw_m_i.detach()
        M_j = self.class_mass

        # Sample-side balancing is normalized over the batch dimension.
        weight_for_W = self._balanced_mass_weight(
            m_i,
            normalize_dim=0,
            gamma=self.sample_gamma,
            alpha=self.sample_alpha,
        )
        # Class-side balancing is normalized over the current class set.
        weight_for_x = self._balanced_mass_weight(
            M_j,
            normalize_dim=1,
            gamma=self.class_gamma,
            alpha=self.class_alpha,
        )

        valid_mask = labels != -1

        with torch.autocast("cuda", enabled=self.fp16):
            norm_x = F.normalize(embeddings, dim=1)
            norm_W = F.normalize(self.weight, dim=1)

            cos_x_for_x = F.linear(norm_x, norm_W.detach())
            cos_x_for_W = F.linear(norm_x.detach(), norm_W)

            logits_x = self._run_logit(cos_x_for_x, labels, raw_m_i)
            logits_W = self._run_logit(cos_x_for_W, labels, raw_m_i)

            logits_x = logits_x * weight_for_x.to(logits_x.dtype)
            logits_W = logits_W * weight_for_W.to(logits_W.dtype)

        ce_x = F.cross_entropy(logits_x, labels, reduction="none", ignore_index=-1)
        ce_W = F.cross_entropy(logits_W, labels, reduction="none", ignore_index=-1)

        loss_x = self._masked_mean(ce_x, valid_mask)
        loss_W = self._masked_mean(ce_W, valid_mask)
        loss = loss_x + loss_W

        with torch.no_grad():
            if valid_mask.any().item():
                valid_labels = labels[valid_mask].long()
                current_m_i = m_i.squeeze(1)[valid_mask]

                sum_m = torch.bincount(
                    valid_labels,
                    weights=current_m_i,
                    minlength=self.num_classes,
                )
                count_m = torch.bincount(valid_labels, minlength=self.num_classes)
                class_mask = count_m > 0

                class_mean_m = sum_m[class_mask] / count_m[class_mask].to(sum_m.dtype)
                self.class_mass[0, class_mask] = (
                    self.class_mass_momentum * self.class_mass[0, class_mask]
                    + (1.0 - self.class_mass_momentum) * class_mean_m
                )
                self.class_mass.clamp_(min=self.eps)

        return loss
