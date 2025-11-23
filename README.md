
# Omni-Zero V4.0 Adaptive

**Omni-Zero** is an experimental JAX/Flax research framework for building
a variable-compute, council-based Transformer with:

- **System 1**: Fast reflexive inference.
- **System 2 (Council)**: Slow, multi-voice simulation.
- **Executive Memory**: Recurrent semantic ring buffer.
- **Neuromodulation**: Plasticity-aware attention and FiLM.
- **Meta-Learning Loop**: Conservative vs aggressive update strategies.
- **Adaptive Gating**: Entropy-triggered System 2 usage.

This repository is organized as a Python package (`omni_zero`) and is
intended for TPU pods or high-end GPU environments running JAX.

> ⚠️ This is research code and not a production system. Expect to modify
> and extend it for your own experiments.

---

## Installation

1. Create and activate a Python environment (3.10+ recommended).
2. Install JAX for your platform (CPU/GPU/TPU). See the official JAX docs.
3. Install the package dependencies:

```bash
pip install -e .
```

You will need:

- `jax`
- `jaxlib`
- `flax`
- `optax`
- `chex`
- `absl-py`

These are declared in `pyproject.toml` and can be installed together
with the editable install above.

---

## Package Layout

```text
omni_zero/
├── __init__.py          # Public API
├── config.py            # Hyperparameters & telemetry types
├── modules.py           # Neuromodulated circuits (PAFA, attention, arbiter)
├── model.py             # Omni-Zero V4.0 Adaptive architecture
├── training.py          # Constitutional meta-learning loop
└── main.py              # Mesh setup & example training step

tests/
└── test_core.py         # Phase 0–style unit test (memory rotation)
```

---

## Quickstart

Once installed, you can run the minimal demo step locally:

```bash
python -m omni_zero.main
```

This will:

- Initialize an `OmniZeroAdaptive` model with the default `Config`.
- Build a dummy batch of world/telemetry tokens.
- Run a single `train_step` and print System 2 activity and loss.

Example output (will vary):

```text
>>> Cortex-X1 Mesh Active: {'data': 1, 'council': 4, 'model': 1}
>>> Initializing Parameters...
>>> Omni-Zero V4.0 Adaptive: ONLINE
>>> Executing Step...
Status: S2 Activity=0.37 | Loss=4.5123
```

---

## Core Concepts

### System 1 vs System 2

- **System 1** (`system_1_pass`):
  - Single forward pass through the executive backbone.
  - Produces logits, value estimate, and a memory write vector.
  - Always runs.

- **System 2 (Council)** (`system_2_council`):
  - Runs multiple persona-parameterized passes in parallel.
  - Each voice performs recursive "ponder" steps over its own thought stream.
  - A `MetaArbiter` fuses the thoughts into a consensus latent.
  - System 1 is then trained to **distill** this consensus.

The top-level `__call__` method in `OmniZeroAdaptive` decides whether
to invoke System 2 based on entropy of the System 1 policy, plus an
optional `force_council` flag.

### Executive Memory

The model maintains a tensor:

```text
executive_memory: [B, memory_slots, embed_dim]
```

On each step, a new `new_thought` vector is written into the **end**
of the ring buffer, while older entries are rotated out. This provides
a simple but explicit episodic memory across timesteps.

### Neuromodulation

Layer-wise plasticity scalars modulate:

- Attention temperature and focus (`NeuromodulatedAttention`).
- Post-attention and post-MLP FiLM gains (`PAFA_FiLM`).

These scalars are part of the meta-learning loop, enabling different
"plasticity profiles" to be explored via conservative vs aggressive
strategies.

### Meta-Learning Loop

`training.py` implements a simplified **conservative vs aggressive**
meta-strategy:

1. The model proposes **two candidate strategies** via the editor head:
   - Conservative (`c_cons`)
   - Aggressive (`c_aggr`)
2. For each strategy, a full gradient step is evaluated hypothetically.
3. The training loop chooses between them based on gradient norms and
   resulting losses, and applies the chosen update to the parameters.
4. An EMA **teacher** tracks a smoothed version of the parameters for
   distillation.

This creates a compact, explicit meta-controller over the learning
process.

---

## Running Tests

```bash
pytest -q
```

or

```bash
python -m pytest
```

The `tests/test_core.py` suite verifies:

- That the executive memory ring buffer rotates correctly.
- That a single `train_step` executes without runtime errors.

---

## Notes & Caveats

- This code is intentionally **opinionated** and experimental.
- Real-world use will require:
  - Proper input tokenization / data pipeline.
  - Careful hyperparameter tuning.
  - TPU mesh configuration specific to your hardware.
- Some parts (e.g., telemetry metrics like attention temperature or
  consensus coherence) are currently stubs and can be wired directly
  from model internals for richer logging.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE)
for details.


---

## PatternWorld Experiment

A minimal **dynamic pattern environment** is provided in `omni_zero/sim.py`:

- Easy phase: linear sequence (+1).
- Hard phase: rule switches (doubling, alternating, etc.).
- When the hidden rule changes, the model experiences a spike in loss,
  and (if configured correctly) the **System 2 Council** should activate
  more frequently.

You can run the end-to-end experiment with:

```bash
python run_experiment.py
```

Watch the console for:

- `Loss` spikes near rule-switch steps (e.g. 30, 60, 90, ...).
- `Sys2?` flipping from `Reflex` to `ACTIVE` around those spikes.
