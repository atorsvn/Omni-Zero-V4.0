
import jax
import jax.numpy as jnp
from jax import nn as jnn
import flax.linen as nn


class PAFA_FiLM(nn.Module):
    """Plasticity-Aware Forward Adapter (Gain Control)."""

    embed_dim: int

    @nn.compact
    def __call__(self, x, plasticity_scalar):
        # plasticity_scalar: [B, 1] or [B, 1, 1]
        if plasticity_scalar.ndim == 1:
            plasticity_scalar = plasticity_scalar[:, None]
        stats = nn.Dense(self.embed_dim * 2, name="film_dense")(plasticity_scalar)
        gamma, beta = jnp.split(stats, 2, axis=-1)
        # Broadcast over time dimension
        if gamma.ndim == 2:
            gamma = gamma[:, None, :]
            beta = beta[:, None, :]
        # 1.0 + gamma ensures identity at low plasticity
        return x * (1.0 + gamma) + beta


class NeuromodulatedAttention(nn.Module):
    """Thalamic Gating: Routing conditioned on plasticity."""

    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x, mask, plasticity_scalar):
        """
        x: [B, T, C]
        mask: [B, 1, 1, T] or None
        plasticity_scalar: [B, 1]
        """
        B, T, C = x.shape
        # Standard QKV
        q = nn.Dense(self.num_heads * self.head_dim, name="q_dense")(x).reshape(
            B, T, self.num_heads, self.head_dim
        )
        k = nn.Dense(self.num_heads * self.head_dim, name="k_dense")(x).reshape(
            B, T, self.num_heads, self.head_dim
        )
        v = nn.Dense(self.num_heads * self.head_dim, name="v_dense")(x).reshape(
            B, T, self.num_heads, self.head_dim
        )

        scores = jnp.einsum("bthd,bshd->bhst", q, k) * (1.0 / jnp.sqrt(self.head_dim))

        # V3.3 Modulation
        modulator = nn.Dense(self.num_heads * 2, name="mod_dense")(plasticity_scalar).reshape(
            B, self.num_heads, 2
        )

        # Temp Scale (Sharpening vs Diffusion)
        temp_scale = 1.0 + jnn.softplus(modulator[:, :, 0])
        scores = scores / temp_scale[:, :, None, None]

        # Focus Bias (Diagonal Enhancement)
        diag_bias = modulator[:, :, 1][:, :, None, None] * jnp.eye(T)[None, None, :, :]
        scores = scores + diag_bias

        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)

        attn = jnn.softmax(scores, axis=-1)
        o = jnp.einsum("bhst,bshd->bthd", attn, v).reshape(B, T, -1)
        out = nn.Dense(C, name="out_proj")(o)
        return out, temp_scale


class MetaArbiter(nn.Module):
    """The Global Workspace: Collapses parallel thoughts into consensus."""

    dim: int

    @nn.compact
    def __call__(self, voices, context_thought):
        """
        voices: [B, K, D]
        context_thought: [B, D]
        """
        query = nn.Dense(self.dim, name="q_proj")(context_thought)[:, None, :]
        keys = nn.Dense(self.dim, name="k_proj")(voices)

        scores = jnp.einsum("bjd,bkd->bk", query, keys) / jnp.sqrt(self.dim)
        weights = jnn.softmax(scores, axis=-1)

        consensus = jnp.einsum("bk,bkd->bd", weights, voices)
        coherence = jnp.max(weights, axis=-1)

        return consensus, weights, coherence
