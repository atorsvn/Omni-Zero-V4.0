
"""
Omni-Zero V4.0 Adaptive
=======================

A research-oriented JAX/Flax implementation of a variable-compute,
council-based Transformer with neuromodulation and meta-learning.

Public entry-points:
- Config (omni_zero.config.Config)
- OmniZeroAdaptive (omni_zero.model.OmniZeroAdaptive)
- create_train_state, train_step (omni_zero.training)
"""
from .config import Config, TelemetryState, MetaStrategy
from .model import OmniZeroAdaptive
from .training import create_train_state, train_step

__all__ = [
    "Config",
    "TelemetryState",
    "MetaStrategy",
    "OmniZeroAdaptive",
    "create_train_state",
    "train_step",
]
