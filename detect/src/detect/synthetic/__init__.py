"""Calibrated synthetic-run generation.

Wires NSE-derived CalibrationParams into synthetic_market_sim config
without modifying the simulator package itself.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns. Quantitative Finance, 1(2), 223-236.
"""
from detect.synthetic.calibrated_runner import (  # noqa: F401
    apply_calibration_to_config,
    generate_calibrated_run,
    regenerate_r01_r24,
)

__all__ = [
    "apply_calibration_to_config",
    "generate_calibrated_run",
    "regenerate_r01_r24",
]
