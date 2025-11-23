import jax
import jax.numpy as jnp
import chex

from omni_zero.modules import MetaArbiter, NeuromodulatedAttention


class TestModules(chex.TestCase):
    def test_meta_arbiter_weights_and_consensus(self):
        voices = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])  # [B=1, K=2, D=2]
        context = jnp.array([[0.5, 0.5]])

        arbiter = MetaArbiter(dim=2)
        params = arbiter.init(jax.random.PRNGKey(0), voices, context)
        consensus, weights, coherence = arbiter.apply(params, voices, context)

        chex.assert_shape(weights, (1, 2))
        chex.assert_shape(consensus, (1, 2))
        chex.assert_trees_all_close(coherence, jnp.max(weights, axis=-1))
        chex.assert_trees_all_close(consensus, jnp.sum(weights[..., None] * voices, axis=1))
        chex.assert_trees_all_close(jnp.sum(weights, axis=-1), jnp.ones((1,)))

    def test_neuromodulated_attention_temperature_scaling(self):
        x = jax.random.normal(jax.random.PRNGKey(1), (1, 3, 4))
        plasticity = jnp.ones((1, 1))

        attn = NeuromodulatedAttention(num_heads=2, head_dim=2)
        params = attn.init(jax.random.PRNGKey(2), x, None, plasticity)
        out, temp_scale = attn.apply(params, x, None, plasticity)

        chex.assert_shape(out, x.shape)
        chex.assert_shape(temp_scale, (1, 2))
        self.assertTrue(jnp.all(temp_scale >= 1.0))


if __name__ == "__main__":
    from absl.testing import absltest

    absltest.main()
