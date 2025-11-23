
import jax.numpy as jnp
from dataclasses import dataclass
from typing import NamedTuple, Any

@dataclass
class Config:
    # --- Model Dimensions ---
    vocab_size: int = 32000
    telemetry_vocab_size: int = 256
    embed_dim: int = 1024
    layers: int = 12
    heads: int = 16

    # --- Executive Memory (V3.4) ---
    memory_slots: int = 16

    # --- The Council (V3.7) ---
    num_voices: int = 4
    ponder_steps: int = 5
    debate_rounds: int = 1

    # --- Meta-Learning ---
    base_lr: float = 1e-4
    max_lr_adjust: float = 2.0
    min_lr_adjust: float = 0.01
    ema_decay: float = 0.999

    # --- Adaptive Gating (V4.0) ---
    entropy_threshold: float = 0.8
    lambda_compute: float = 0.01  # Penalty for using System 2

    # --- Sharding (Cortex-X1) ---
    mesh_axes = ("data", "council", "model")


class TelemetryState(NamedTuple):
    last_loss: float
    grad_norm: float
    layer_stability: float
    attention_temp: float
    consensus_coherence: float
    system_2_active: float  # 0.0 or 1.0


class MetaStrategy(NamedTuple):
    global_lr: float
    layer_plasticity: Any  # Array [Layers]
    lambdas: Any           # Array [3]
