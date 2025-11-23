
import jax
import jax.numpy as jnp

from omni_zero.config import Config
from omni_zero.model import OmniZeroAdaptive
from omni_zero.training import create_train_state, train_step
from omni_zero.sim import PatternWorld
from omni_zero.utils import TelemetryTokenizer


def main():
    # 1. Setup
    config = Config(
        vocab_size=100,      # Small vocab for math task
        layers=4,            # Smaller model for local testing
        heads=4,
        embed_dim=64,
        memory_slots=8,
        num_voices=2,        # 2 Voices for smaller hardware
        ponder_steps=3,
        entropy_threshold=0.6,  # Sensitivity to confusion
    )

    world = PatternWorld(vocab_size=config.vocab_size, switch_every=30)
    tokenizer = TelemetryTokenizer(vocab_size=config.telemetry_vocab_size)

    # Init Model
    rng = jax.random.PRNGKey(1337)
    rng, init_rng = jax.random.split(rng)

    model = OmniZeroAdaptive(config)

    # Dummy init args
    dummy_w = jnp.zeros((1, 64), dtype=jnp.int32)
    dummy_t = jnp.zeros((1, 32, 4), dtype=jnp.int32)
    dummy_m = jnp.zeros((1, config.memory_slots, config.embed_dim), dtype=jnp.float32)
    dummy_p = jnp.ones((config.layers,), dtype=jnp.float32)

    params = model.init(init_rng, dummy_w, dummy_t, dummy_m, dummy_p, init_rng)["params"]
    state = create_train_state(model, params, config)

    # Init Env
    rng, env_rng = jax.random.split(rng)
    env_state, obs = world.reset(env_rng)

    # Telemetry History (buffer for the agent to see its own stats)
    tel_buffer = [
        {"loss": 0.0, "grad_norm": 0.0, "entropy": 0.0, "lr": config.base_lr}
        for _ in range(32)
    ]

    print(f"{'Step':<5} | {'Rule':<5} | {'Loss':<8} | {'Sys2?':<8} | {'Action':<6} | {'Reward':<6}")
    print("-" * 70)

    # 2. Training Loop (simple on-policy teacher-forced prediction)
    for step in range(200):
        rng, step_rng = jax.random.split(rng)

        # Prepare Inputs (batch size = 1)
        w_tokens = obs[None, :]  # [1, T]
        t_tokens = tokenizer.batch_encode(tel_buffer)[None, :, :]  # [1, T_t, 4]

        # Compute "true" next token target from env state (teacher forcing)
        def get_target(s):
            if s.current_rule == 0:
                return (s.last_token + 1) % 100
            if s.current_rule == 1:
                return (s.last_token * 2) % 100
            return (s.last_token + (1 if s.time_step % 2 == 0 else -1)) % 100

        true_target = get_target(env_state)

        batch = {
            "world_tokens": w_tokens,
            "telemetry_tokens": t_tokens,
            "target_action": jnp.array([true_target]),
            "target_telemetry": jnp.zeros((1,), dtype=jnp.int32),
        }

        state, metrics = train_step(state, batch, step_rng)

        # Approximate entropy proxy: just reuse loss for demonstration
        entropy_proxy = float(metrics.last_loss)

        # Update Telemetry Buffer
        tel_buffer.pop(0)
        tel_buffer.append(
            {
                "loss": float(metrics.last_loss),
                "grad_norm": float(metrics.grad_norm),
                "entropy": entropy_proxy,
                "lr": float(getattr(getattr(state, "tx", None), "learning_rate", config.base_lr))
                if hasattr(state, "tx")
                else config.base_lr,
            }
        )

        # Here we *should* pick the model's action using its logits.
        # For simplicity in this compact script, we use the true target
        # as the action the agent "takes", to focus on the gating behavior.
        action = int(true_target)

        env_state, obs, reward, _ = world.step(env_state, action, step_rng)

        sys2_status = "ACTIVE" if metrics.system_2_active > 0.5 else "Reflex"

        print(
            f"{step:<5} | {env_state.current_rule:<5} | "
            f"{metrics.last_loss:8.4f} | {sys2_status:<8} | "
            f"{action:<6} | {reward:6.2f}"
        )


if __name__ == "__main__":
    main()
