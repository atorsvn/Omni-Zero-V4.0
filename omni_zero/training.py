
import jax
import jax.numpy as jnp
from jax import lax, tree_util
import optax
from flax.training import train_state

from .config import TelemetryState, MetaStrategy, Config


class AdaptiveTrainState(train_state.TrainState):
    telemetry: TelemetryState
    current_plasticity: jax.Array
    teacher_params: object
    executive_memory: jax.Array


def create_train_state(model, params, config: Config):
    teacher = tree_util.tree_map(lambda x: x, params)
    tx = optax.adamw(config.base_lr)
    return AdaptiveTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        telemetry=TelemetryState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        current_plasticity=jnp.ones((config.layers,)),
        teacher_params=teacher,
        executive_memory=jnp.zeros((1, config.memory_slots, config.embed_dim)),
    )


@jax.jit
def train_step(state: AdaptiveTrainState, batch, rng):
    rng, _ = jax.random.split(rng)

    # --- Counterfactual Meta-Loop ---

    def compute_branch(params, strategy: MetaStrategy):
        # Freeze meta-weights for loss
        l_rl, l_self, l_dist = [lax.stop_gradient(x) for x in strategy.lambdas]

        def loss_fn(p):
            (
                logits,
                val,
                pred_tel,
                mem,
                _,
                _temp,
                _coh,
                s2_active,
            ) = state.apply_fn(
                {"params": p},
                batch["world_tokens"],
                batch["telemetry_tokens"],
                state.executive_memory,
                state.current_plasticity,
                rng,
                force_council=False,
            )

            # Base Losses
            L_RL = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch["target_action"]
            ).mean()

            L_Self = optax.softmax_cross_entropy_with_integer_labels(
                pred_tel, batch["target_telemetry"]
            ).mean()

            # Distill
            diff = tree_util.tree_map(lambda x, y: jnp.sum((x - y) ** 2), p, state.teacher_params)
            L_Distill = sum(tree_util.tree_leaves(diff))

            # Compute Penalty
            L_Compute = 0.01 * jnp.mean(s2_active)

            total = (l_rl * L_RL) + (l_self * L_Self) + (l_dist * L_Distill) + L_Compute
            return total, (L_RL, mem, val, s2_active)

        grads, (raw_loss, thought, v, active) = jax.grad(loss_fn, has_aux=True)(params)

        # Apply Plasticity (Global only for brevity, real impl uses layer masks)
        grads = tree_util.tree_map(lambda g: g * strategy.global_lr, grads)
        return grads, optax.global_norm(grads), raw_loss, thought, active

    # 1. Get Strategies from a clean forward pass
    _, _, _, _, (cand_cons, cand_aggr), _, _, _ = state.apply_fn(
        {"params": state.params},
        batch["world_tokens"],
        batch["telemetry_tokens"],
        state.executive_memory,
        state.current_plasticity,
        rng,
    )

    def extract(c: MetaStrategy) -> MetaStrategy:
        return MetaStrategy(
            c.global_lr.mean(),
            c.layer_plasticity.mean(0),
            c.lambdas.mean(0),
        )

    s_cons = extract(cand_cons)
    s_aggr = extract(cand_aggr)

    # 2. Evaluate Futures
    g_c, n_c, l_c, t_c, a_c = compute_branch(state.params, s_cons)
    g_a, n_a, l_a, t_a, a_a = compute_branch(state.params, s_aggr)

    # 3. Arbitrate
    is_stable = n_a < (n_c * 1.5)

    final_grads = tree_util.tree_map(
        lambda x, y: jnp.where(is_stable, y, x), g_c, g_a
    )
    final_plast = jnp.where(is_stable, s_aggr.layer_plasticity, s_cons.layer_plasticity)
    final_thought = jnp.where(is_stable, t_a, t_c)

    # 4. Update
    state = state.apply_gradients(grads=final_grads)

    # Memory Update (Ring Buffer)
    curr_mem = state.executive_memory
    new_mem = jnp.concatenate([curr_mem[:, 1:, :], final_thought[:, None, :]], axis=1)

    # Teacher Update
    ema = 0.999
    new_teacher = tree_util.tree_map(
        lambda s, t: ema * t + (1.0 - ema) * s, state.params, state.teacher_params
    )

    new_tel = TelemetryState(
        last_loss=jnp.where(is_stable, l_a, l_c),
        grad_norm=jnp.where(is_stable, n_a, n_c),
        layer_stability=jnp.mean(final_plast),
        attention_temp=0.0,  # Placeholder
        consensus_coherence=0.0,  # Placeholder
        system_2_active=jnp.where(is_stable, a_a, a_c).mean(),
    )

    return (
        state.replace(
            telemetry=new_tel,
            current_plasticity=final_plast,
            teacher_params=new_teacher,
            executive_memory=new_mem,
        ),
        new_tel,
    )
