
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

from .config import Config
from .model import OmniZeroAdaptive
from .training import create_train_state, train_step


def setup_mesh(num_voices: int = 4):
    devices = jax.devices()
    n = len(devices)
    if n < num_voices:
        print(f"Warning: Only {n} devices. Emulation will overlap.")
        mesh = Mesh(mesh_utils.create_device_mesh((1, 1, 1)), ("data", "council", "model"))
    else:
        # Heuristic: prioritize council isolation
        c = num_voices
        remaining = n // c
        m = int(remaining ** 0.5) or 1
        d = max(remaining // max(m, 1), 1)
        mesh = Mesh(mesh_utils.create_device_mesh((d, c, m)), ("data", "council", "model"))
    return mesh


def main():
    config = Config()
    mesh = setup_mesh(config.num_voices)

    print(f">>> Cortex-X1 Mesh Active: {mesh.shape}")

    # Init model
    model = OmniZeroAdaptive(config)
    rng = jax.random.PRNGKey(42)

    # Dummy inputs
    d_w = jnp.ones((1, 64), dtype=jnp.int32)
    d_t = jnp.ones((1, 32, 4), dtype=jnp.int32)
    d_m = jnp.zeros((1, config.memory_slots, config.embed_dim), dtype=jnp.float32)
    d_p = jnp.ones((config.layers,), dtype=jnp.float32)

    print(">>> Initializing Parameters...")
    params = model.init(rng, d_w, d_t, d_m, d_p, rng)["params"]
    state = create_train_state(model, params, config)

    print(">>> Omni-Zero V4.0 Adaptive: ONLINE")

    # Mock batch
    batch = {
        "world_tokens": jnp.ones((4, 64), dtype=jnp.int32),
        "telemetry_tokens": jnp.ones((4, 32, 4), dtype=jnp.int32),
        "target_action": jnp.ones((4,), dtype=jnp.int32),
        "target_telemetry": jnp.ones((4,), dtype=jnp.int32),
    }

    print(">>> Executing Step...")
    state, metrics = train_step(state, batch, rng)

    print(
        f"Status: S2 Activity={metrics.system_2_active:.2f} | "
        f"Loss={metrics.last_loss:.4f}"
    )


if __name__ == "__main__":
    main()
