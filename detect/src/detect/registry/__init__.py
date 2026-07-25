"""Experiment registry - loads YAML configs and persists trained artifacts.

Cont, R. (2001). Empirical properties of asset returns. Quantitative Finance, 1(2), 223-236.
"""
from detect.registry.experiment import ExperimentConfig  # noqa: F401
from detect.registry.checkpoints import save_experiment, list_experiments, load_experiment  # noqa: F401

__all__ = ["ExperimentConfig", "save_experiment", "list_experiments", "load_experiment"]
