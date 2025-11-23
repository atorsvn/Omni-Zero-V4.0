
import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple


class EnvState(NamedTuple):
    time_step: int
    current_rule: int  # 0: Linear (+1), 1: Doubling (*2), 2: Alternating (+1, -1)
    last_token: int
    seq_history: jax.Array  # [T] context window


class PatternWorld:
    """
    A JAX-native environment that switches rules to force 'Surprise' (Entropy).
    """

    def __init__(self, vocab_size: int = 100, switch_every: int = 50, history_len: int = 64):
        self.vocab_size = vocab_size
        self.switch_every = switch_every
        self.history_len = history_len

    def reset(self, rng) -> Tuple[EnvState, jax.Array]:
        # Start with Linear Rule
        initial_token = jax.random.randint(rng, (), 1, 10)
        state = EnvState(
            time_step=0,
            current_rule=0,
            last_token=int(initial_token),
            seq_history=jnp.zeros((self.history_len,), dtype=jnp.int32),
        )
        return state, self._get_obs(state)

    def step(self, state: EnvState, action: int, rng) -> Tuple[EnvState, jax.Array, float, bool]:
        # 1. Determine Correct Answer based on Current Rule
        # Rule 0: Linear (n+1)
        # Rule 1: Doubling (n*2)
        # Rule 2: Alternating (n + (-1)^t)

        def rule_linear(x, t):
            return (x + 1) % self.vocab_size

        def rule_double(x, t):
            return (x * 2) % self.vocab_size

        def rule_alt(x, t):
            return (x + (1 if t % 2 == 0 else -1)) % self.vocab_size

        target = jax.lax.switch(
            state.current_rule,
            [rule_linear, rule_double, rule_alt],
            state.last_token,
            state.time_step,
        )

        # 2. Calculate Reward (Binary for simplicity, or negative distance)
        reward = jnp.where(action == target, 1.0, -0.1)

        # 3. Evolve Environment
        new_time = state.time_step + 1

        # Check if we should switch rules (The "Surprise" Event)
        should_switch = (new_time % self.switch_every) == 0
        new_rule = jnp.where(should_switch, (state.current_rule + 1) % 3, state.current_rule)

        # Update History
        new_hist = jnp.roll(state.seq_history, -1)
        new_hist = new_hist.at[-1].set(target)

        new_state = EnvState(
            time_step=int(new_time),
            current_rule=int(new_rule),
            last_token=int(target),
            seq_history=new_hist,
        )

        return new_state, self._get_obs(new_state), float(reward), False

    def _get_obs(self, state: EnvState) -> jax.Array:
        # The agent sees the sequence history
        return state.seq_history
