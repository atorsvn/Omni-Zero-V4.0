
import jax
import jax.numpy as jnp
from jax import lax, random, vmap
from jax.sharding import PartitionSpec as P
import flax.linen as nn

from .config import Config, MetaStrategy
from .modules import PAFA_FiLM, NeuromodulatedAttention, MetaArbiter


class ExecutiveBlock(nn.Module):
    features: int
    num_heads: int
    layer_id: int

    @nn.compact
    def __call__(self, x, mask, layer_plasticity):
        """
        x: [B, T, D]
        layer_plasticity: [B, L] or [L]
        """
        if layer_plasticity.ndim == 1:
            p_val = jnp.broadcast_to(layer_plasticity[self.layer_id], (x.shape[0], 1))
        else:
            p_val = layer_plasticity[:, self.layer_id : self.layer_id + 1]

        # Attention + Thalamic Gating
        x_norm = nn.RMSNorm(name="attn_norm")(x)
        attn_mod = NeuromodulatedAttention(
            num_heads=self.num_heads,
            head_dim=self.features // self.num_heads,
            name="attn",
        )
        x_attn, temp = attn_mod(x_norm, mask, p_val)
        x = x + x_attn

        # PAFA (Pre-MLP)
        x = PAFA_FiLM(self.features, name="film_attn")(x, p_val)

        # MLP
        h = nn.RMSNorm(name="mlp_norm")(x)
        h = nn.Dense(self.features * 4, name="mlp_fc1")(h)
        h = nn.gelu(h)
        h = nn.Dense(self.features, name="mlp_fc2")(h)
        x = x + h

        # PAFA (Post-MLP)
        x = PAFA_FiLM(self.features, name="film_mlp")(x, p_val)

        return x, temp


class OmniZeroAdaptive(nn.Module):
    """Omni-Zero V4.0 Adaptive architecture."""

    config: Config

    def setup(self):
        cfg = self.config
        self.world_embed = nn.Embed(cfg.vocab_size, cfg.embed_dim, name="world_embed")
        self.telemetry_embed = nn.Embed(
            cfg.telemetry_vocab_size, cfg.embed_dim, name="telemetry_embed"
        )
        self.memory_proj = nn.Dense(cfg.embed_dim, name="memory_proj")

        # V3.7 Personas (sharded on council axis via training loop)
        self.persona_embeds = self.param(
            "personas",
            nn.initializers.normal(0.02),
            (cfg.num_voices, cfg.embed_dim),
        )

        self.layers = [
            ExecutiveBlock(cfg.embed_dim, cfg.heads, i, name=f"block_{i}")
            for i in range(cfg.layers)
        ]
        self.norm = nn.RMSNorm(name="final_norm")

        self.policy_head = nn.Dense(cfg.vocab_size, name="policy_head")
        self.value_head = nn.Dense(1, name="value_head")
        self.introspection_head = nn.Dense(
            cfg.telemetry_vocab_size, name="introspection_head"
        )
        self.memory_write_head = nn.Sequential(
            [nn.Dense(cfg.embed_dim), nn.tanh], name="memory_write_head"
        )

        self.meta_arbiter = MetaArbiter(cfg.embed_dim, name="meta_arbiter")
        # (global_lr + layer_plasticity[L] + lambdas[3]) * 2 for conservative/aggressive
        self.editor_head = nn.Sequential(
            [
                nn.Dense(256),
                nn.swish,
                nn.Dense((1 + cfg.layers + 3) * 2),
            ],
            name="editor_head",
        )

    def _backbone(self, x, mask, plasticity, persona_bias=None):
        if persona_bias is not None:
            x = x + persona_bias[:, None, :]

        temps = []
        for block in self.layers:
            x, t = block(x, mask, plasticity)
            temps.append(t.mean())
        x = self.norm(x)
        return x, jnp.mean(jnp.stack(temps))

    def _prepare_input(self, w_tok, t_tok, mem, sim_mode: bool = False):
        """
        w_tok: [B, T_w] or [B, T_w, D] in sim_mode
        t_tok: [B, T_t, T_inner] or [B, T_t]
        mem:   [B, M, D]
        """
        # Flatten telemetry tokens if they have inner structure
        if t_tok.ndim == 3:
            B, T_t, T_in = t_tok.shape
            t_flat = t_tok.reshape(B, T_t * T_in)
        else:
            t_flat = t_tok
        t_emb = self.telemetry_embed(t_flat)  # [B, T_t_flat, D]
        m_emb = self.memory_proj(mem)        # [B, M, D]

        if sim_mode:
            # w_tok is already an embedding [B, 1, D]
            w_emb = w_tok
        else:
            w_emb = self.world_embed(w_tok)  # [B, T_w, D]

        x = jnp.concatenate([m_emb, t_emb, w_emb], axis=1)
        offset = m_emb.shape[1] + t_emb.shape[1]
        return x, offset

    def system_1_pass(self, w, t, m, p):
        """Fast Path: No simulation, no voices."""
        x, offset = self._prepare_input(w, t, m)
        x, temp = self._backbone(x, None, p)

        latent = x[:, -1, :]
        latent_self = x[:, offset - 1, :]

        logits = self.policy_head(latent)
        value = self.value_head(latent)
        mem_write = self.memory_write_head(latent)

        return logits, value, mem_write, latent, latent_self, temp

    def system_2_council(self, w, t, m, p, rng):
        """Slow Path: Parallel Voice Simulation + Arbitration."""
        cfg = self.config

        # 1. Prepare Personas (constraint injection for Cortex-X1)
        personas = self.persona_embeds
        try:
            personas = lax.with_sharding_constraint(personas, P("council", "model"))
        except RuntimeError:
            # Tests and CPU-only runs may not have an active mesh; skip sharding.
            pass

        def single_voice(voice_idx, sub_rng):
            persona = personas[voice_idx]
            batch_persona = jnp.repeat(persona[None, :], w.shape[0], axis=0)

            # Root pass
            x, _ = self._prepare_input(w, t, m)
            x, _ = self._backbone(x, None, p, batch_persona)
            root_thought = x[:, -1, :]

            # Recursive Simulation
            def scan_fn(carrier, _):
                curr, _, rng_inner = carrier
                step_rng, new_rng = random.split(rng_inner)

                # Sim step: feed current thought as "world embedding"
                sim_x, _ = self._prepare_input(curr[:, None, :], t, m, sim_mode=True)
                sim_x, _ = self._backbone(sim_x, None, p, batch_persona)
                next_thought = sim_x[:, -1, :]
                val = self.value_head(next_thought)
                return (next_thought, val, new_rng), val

            (final_thought, final_val, _), _ = lax.scan(
                scan_fn,
                (root_thought, self.value_head(root_thought), sub_rng),
                None,
                length=cfg.ponder_steps,
            )
            return final_thought, final_val, root_thought

        rngs = random.split(rng, cfg.num_voices)
        indices = jnp.arange(cfg.num_voices)

        c_thoughts, c_vals, c_roots = vmap(single_voice)(indices, rngs)

        # [Voices, B, D] -> [B, Voices, D]
        c_thoughts = jnp.transpose(c_thoughts, (1, 0, 2))
        c_roots = jnp.transpose(c_roots, (1, 0, 2))

        # 3. Arbitration
        context = jnp.mean(c_roots, axis=1)
        consensus, weights, coherence = self.meta_arbiter(c_thoughts, context)

        consensus_logits = self.policy_head(consensus)
        consensus_val = jnp.mean(jnp.transpose(c_vals, (1, 0, 2)), axis=1)

        return consensus_logits, consensus_val, consensus, coherence

    def __call__(self, w, t, m, p, rng, force_council: bool = False):
        """
        w: world tokens [B, T_w]
        t: telemetry tokens [B, T_t] or [B, T_t, T_inner]
        m: executive memory [B, M, D]
        p: layer plasticity [L] or [B, L]
        """
        # 1. System 1 Reflex
        s1_logits, s1_val, s1_mem, s1_lat, s1_self, temp = self.system_1_pass(w, t, m, p)

        # 2. Entropy Check
        probs = nn.softmax(s1_logits, axis=-1)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-9), axis=-1).mean()
        should_wake = (entropy > self.config.entropy_threshold) | force_council

        # 3. Adaptive Gating
        def run_s2(_):
            s2_logits, s2_val, s2_lat, coh = self.system_2_council(w, t, m, p, rng)
            s2_mem = self.memory_write_head(s2_lat)
            return s2_logits, s2_val, s2_mem, s2_lat, coh, 1.0

        def run_s1(_):
            zeros = jnp.zeros((w.shape[0],), dtype=jnp.float32)
            return s1_logits, s1_val, s1_mem, s1_lat, zeros, 0.0

        if self.config.num_voices == 0:
            final_outputs = run_s1(None)
        else:
            final_outputs = lax.cond(should_wake, run_s2, run_s1, operand=None)

        final_logits, final_val, final_mem, final_lat, coh, s2_active = final_outputs

        # 4. Meta-Strategy Generation (From Self Latent)
        meta_raw = self.editor_head(s1_self)
        mean, delta = jnp.split(meta_raw, 2, axis=-1)
        cons_raw = mean - jnp.abs(delta)
        aggr_raw = mean + jnp.abs(delta)

        c_cons = self._project(cons_raw)
        c_aggr = self._project(aggr_raw)

        # Introspection
        pred_tel = self.introspection_head(s1_self)

        return (
            final_logits,
            final_val,
            pred_tel,
            final_mem,
            (c_cons, c_aggr),
            temp,
            coh,
            s2_active,
        )

    def _project(self, raw):
        """
        raw: [B, 1 + L + 3]
        """
        cfg = self.config
        # global lr
        g_raw = raw[:, 0]
        l_raw = raw[:, 1 : 1 + cfg.layers]
        lam_raw = raw[:, 1 + cfg.layers :]

        g = cfg.min_lr_adjust + (cfg.max_lr_adjust - cfg.min_lr_adjust) * nn.sigmoid(g_raw)
        l = 2.0 * nn.sigmoid(l_raw)
        lam = 0.1 + 2.0 * nn.sigmoid(lam_raw)
        return MetaStrategy(g, l, lam)
