"""Focal loss for edge-level prediction with extreme class imbalance.

Lin, T-Y., Goyal, P., Girshick, R., He, K., Dollar, P. (2017).
"Focal Loss for Dense Object Detection." ICCV.

The edge-level GraphSAGE task is ~0.85% positive (manipulator edges among
all trader-trader edges per run). Weighted BCE handles the imbalance
linearly via pos_weight; focal loss handles it non-linearly via
(1 - p_t)^gamma — i.e. it down-weights easy (high-confidence) examples
and focuses gradient on hard ones.

Empirically this gives sharper per-edge separation in extreme-imbalance
regimes. The trader projection still aggregates the same way (max + top3)
but operates on a more discriminative score distribution.

Usage:
    criterion = FocalLoss(alpha=0.85, gamma=2.0)
    loss = criterion(logits, targets)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss.

    Args
    ----
    alpha
        Class-balance weight for the positive class in [0, 1]. The negative
        class is weighted (1 - alpha). Authors recommend 0.25 for object
        detection (~few positives per image); for our regime (~0.85%
        positives) values closer to 0.85 (=1 - 0.15) match what the
        weighted-BCE pos_weight already does.
    gamma
        Focusing parameter. gamma=0 reduces to standard BCE; gamma=2 is
        the paper's recommended default and works well as a starting point
        on the MSA cohort.
    reduction
        "mean" (default), "sum", or "none".
    """

    def __init__(self, alpha: float = 0.85, gamma: float = 2.0,
                 reduction: str = "mean") -> None:
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """logits: any shape; targets: same shape, values in {0, 1} (float)."""
        if logits.shape != targets.shape:
            raise ValueError(
                f"shape mismatch: logits={logits.shape} targets={targets.shape}"
            )
        # bce_per_example is the non-reduced BCE; same shape as inputs.
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )
        # p_t = probability of the true class:
        #   if y=1: p_t = sigmoid(x)
        #   if y=0: p_t = 1 - sigmoid(x)
        p = torch.sigmoid(logits)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        # alpha_t balances positive vs negative class weighting.
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        # (1 - p_t)^gamma down-weights easy examples (those where p_t -> 1).
        modulating = (1.0 - p_t) ** self.gamma
        loss = alpha_t * modulating * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def make_loss(name: str, *, pos_weight: Optional[float] = None,
              alpha: float = 0.85, gamma: float = 2.0) -> nn.Module:
    """Factory for the two losses the trainer supports.

    name="bce" → standard weighted BCE (M3 / M3+ baseline).
    name="focal" → FocalLoss with the given alpha and gamma.
    """
    if name == "bce":
        if pos_weight is None:
            return nn.BCEWithLogitsLoss()
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(pos_weight))
        )
    if name == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
    raise ValueError(f"unknown loss '{name}'. Expected 'bce' or 'focal'.")
