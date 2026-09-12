"""
Focal loss with class weights + label smoothing for heavy class imbalance.

Focal loss down-weights easy, well-classified samples and focuses learning on
hard/minority classes; class weights further counter imbalance; a little label
smoothing improves calibration and generalisation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        n, k = logits.shape

        # per-sample class weight
        if self.weight is not None:
            w = self.weight[target]
        else:
            w = torch.ones(n, device=logits.device)

        pt = p.gather(1, target.unsqueeze(1)).squeeze(1).clamp_(1e-6, 1.0)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        focal = (1.0 - pt) ** self.gamma

        # hard-target focal CE
        loss_hard = -focal * logpt

        if self.label_smoothing > 0:
            # uniform smoothing term
            smooth = -logp.mean(dim=1)
            ls = self.label_smoothing
            loss = (1 - ls) * loss_hard + ls * focal * smooth
        else:
            loss = loss_hard

        return (w * loss).mean()


def mixup_focal(loss_fn, logits, y_a, y_b, lam):
    """Focal loss under mixup: convex combination of the two targets."""
    return lam * loss_fn(logits, y_a) + (1.0 - lam) * loss_fn(logits, y_b)
