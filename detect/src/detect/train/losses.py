"""Loss helpers - weighted BCE pos-weight resolver, focal loss factory.

Used by Rung 4 (torch). Rung 3's XGBoost handles imbalance through
``sample_weight`` instead, computed by the same helper here.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.

Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollar, P. (2017). Focal
Loss for Dense Object Detection. ICCV. (Source of focal loss.)
"""
from __future__ import annotations

import numpy as np


def auto_pos_weight(y) -> float:
    """``pos_weight = N_negative / N_positive``. Falls back to 1.0 when degenerate."""
    y_arr = np.asarray(y).astype(int)
    pos = int((y_arr == 1).sum())
    neg = int((y_arr == 0).sum())
    if pos == 0 or neg == 0:
        return 1.0
    return float(neg) / float(pos)


def auto_sample_weight(y, pos_weight: "float | None" = None) -> np.ndarray:
    """Sample-weight vector for tabular trainers (xgboost / sklearn)."""
    pw = float(pos_weight) if pos_weight is not None else auto_pos_weight(y)
    y_arr = np.asarray(y).astype(int)
    return np.where(y_arr == 1, pw, 1.0).astype(np.float32)


def focal_loss_factory(gamma: float = 2.0, alpha: float = 0.25):
    """Return a torch loss callable for binary focal loss.

    Implemented lazily to avoid importing torch at module load.
    Reference: Lin et al. 2017.
    """
    def _focal_loss(logits, targets):  # type: ignore[no-untyped-def]
        import torch
        import torch.nn.functional as F
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** gamma * ce
        return loss.mean()
    return _focal_loss
