"""Feature engineering layer.

Cont, R. (2001). Empirical properties of asset returns. Quantitative Finance, 1(2), 223-236.
"""
from detect.features.edge_engineered import compute_edge_features_v1  # noqa: F401

__all__ = ["compute_edge_features_v1"]
