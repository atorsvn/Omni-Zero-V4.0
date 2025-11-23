
import jax
import jax.numpy as jnp
import chex
from absl.testing import absltest

from omni_zero.config import Config
from omni_zero.model import OmniZeroAdaptive
from omni_zero.training import create_train_state, train_step


class TestV4Core(chex.TestCase):
    def setUp(self):
        self.config = Config(
            vocab_size=100,
            embed_dim=16,
            layers=2,
            heads=2,
            memory_slots=4,
            num_voices=0,
            ponder_steps=2,
        )
        self.model = OmniZeroAdaptive(self.config)
        self.rng = jax.random.PRNGKey(0)

        d_w = jnp.ones((1, 10), dtype=jnp.int32)
        d_t = jnp.ones((1, 5, 4), dtype=jnp.int32)
        d_m = jnp.zeros((1, 4, 16), dtype=jnp.float32)
        d_p = jnp.ones((2,), dtype=jnp.float32)

        self.params = self.model.init(self.rng, d_w, d_t, d_m, d_p, self.rng)["params"]
        self.state = create_train_state(self.model, self.params, self.config)

    def test_memory_rotation(self):
        mem = self.state.executive_memory
        mem = mem.at[:, 0, :].set(10.0)
        state = self.state.replace(executive_memory=mem)

        batch = {
            "world_tokens": jnp.ones((1, 10), dtype=jnp.int32),
            "telemetry_tokens": jnp.ones((1, 5, 4), dtype=jnp.int32),
            "target_action": jnp.ones((1,), dtype=jnp.int32),
            "target_telemetry": jnp.ones((1,), dtype=jnp.int32),
        }

        new_state, _ = train_step(state, batch, self.rng)

        chex.assert_trees_all_close(
            new_state.executive_memory[:, 0, :], jnp.zeros_like(mem[:, 0, :])
        )
        self.assertGreater(
            jnp.abs(new_state.executive_memory[:, -1, :]).sum(), 0.0
        )


if __name__ == "__main__":
    absltest.main()
