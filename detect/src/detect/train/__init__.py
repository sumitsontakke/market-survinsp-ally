"""Training entrypoint and shared primitives.

Cont, R. (2001). Empirical properties of asset returns. Quantitative Finance, 1(2), 223-236.
"""
from detect.train.projection import project_edge_probs_to_traders  # noqa: F401

__all__ = ["project_edge_probs_to_traders"]
