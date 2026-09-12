"""
Training-time augmentation for microbiome vectors (applied on TRANSFORMED input).

These make the encoder rely on ROBUST, redundant microbial signal instead of a
few study-specific taxa, which is exactly what we need for cross-study
generalisation.

  * feature_dropout : randomly zero a fraction of features (species) per sample
  * gaussian_noise  : small additive noise
  * mixup           : convex combination of two samples + their labels

All operate on torch tensors already in CLR/standardised space.
"""
import torch
import numpy as np


def feature_dropout(x: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return x
    mask = (torch.rand_like(x) > p).float()
    return x * mask


def gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return x
    return x + torch.randn_like(x) * std


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Return mixed_x, y_a, y_b, lam. If alpha<=0, no mixup."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1.0 - lam) * x[perm]
    return mixed, y, y[perm], lam
