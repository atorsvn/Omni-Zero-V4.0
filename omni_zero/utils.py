
import jax.numpy as jnp
from typing import Dict, List


class TelemetryTokenizer:
    """
    Translates continuous internal states (Loss, Gradients, Entropy)
    into discrete tokens for the Introspection Engine.
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        # We define ranges for normalization: [Min, Max]
        self.ranges = {
            "loss": (0.0, 10.0),
            "grad_norm": (0.0, 5.0),
            "entropy": (0.0, 3.5),  # Log(Vocab) approx
            "lr": (0.0, 1e-3),
        }

    def encode(self, metrics: Dict[str, float]) -> jnp.ndarray:
        """
        Input: Dict of raw floats.
        Output: Array of token IDs [4] (Loss, Grad, Ent, LR).
        """
        tokens = []
        order = ["loss", "grad_norm", "entropy", "lr"]

        for key in order:
            val = metrics.get(key, 0.0)
            low, high = self.ranges[key]

            # Normalize 0-1
            norm = (val - low) / (high - low + 1e-9)
            norm = jnp.clip(norm, 0.0, 1.0)

            # Quantize
            token_id = (norm * (self.vocab_size - 1)).astype(jnp.int32)
            tokens.append(token_id)

        return jnp.array(tokens, dtype=jnp.int32)

    def batch_encode(self, history: List[Dict[str, float]]) -> jnp.ndarray:
        """
        Processes a time-series of metrics for the Self-Context window.
        Output: [Seq_Len, 4]
        """
        return jnp.stack([self.encode(m) for m in history])
