---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: py:percent,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
---

# 06 — Simple Training Tutorial

This mini-notebook shows an end‑to‑end training loop. Notice that it resembles
plain pytorch in the sense that you need to write your own train_Step and eval_step.
We will see:

- How to define a `SequentialOrchestrator` over a sparse `LayerMap`.
- How to update the model via `.backward(...)` used as **pseudo‑gradients** for Optax.
- How to define train_steps, eval_steps and update_steps.

The goal here is **clarity**: well‑ordered cells, consistent dtypes/PRNG usage,
and inline comments explaining each step. We are not taking full advantage of jax for now.
For example here we never call Jit on any function. Also, the dynamics `orchestrator.step/step_inference`
is well suited for `jax.lax.scan`, but here we will just use a regular python-side for loop.

```python
# --- Imports ---------------------------------------------------------------
import time
from typing import Any

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from bionet.orchestrators.sequential import SequentialOrchestrator
from bionet.modules.fully_connected import FrozenFullyConnected, FullyConnected
from bionet.modules.recurrent import RecurrentDiscrete
from bionet.modules.input_output import OutputLayer
from bionet.layer_maps.sparse import LayerMap
from bionet.states.sequential import SequentialState
```


## Utilities (metrics & summaries)

Small helpers used for monitoring. Labels are OVA ±1 and predictions are
decoded via `argmax` over the output scores.


```python
def batch_accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Accuracy with ±1 OVA labels (class = argmax along last dim)."""
    y_true_idx = jnp.argmax(y_true, axis=-1)
    y_pred_idx = jnp.argmax(y_pred, axis=-1)
    return float(jnp.mean(y_true_idx == y_pred_idx))


def print_state_summary(s_out: jnp.ndarray, header: str = "Output state") -> None:
    """Quick shape/range summary for an output buffer."""
    smin = float(jnp.min(s_out))
    smax = float(jnp.max(s_out))
    smu = float(jnp.mean(s_out))
    sstd = float(jnp.std(s_out))
    print(
        f"{header}: shape={tuple(s_out.shape)}, range=({smin:.3f},{smax:.3f}), mean={smu:.3f}, std={sstd:.3f}"
    )
```


## Dataset: Binary Prototypes.

This object defines the data that we will be training our model on. It is a very simple task.
The object is a compact generator that creates:
- `K` prototype vectors in `{−1,+1}^D`.
- A dataset of size `N` by corrupting prototypes with random bit-flips with probability `p_noise`.
- ±1 labels (`+1` on the correct class, `−1` elsewhere), that indicate which prototype the data is coming from.


```python
class PrototypeData:
    """Binary prototype dataset with OVA ±1 labels."""

    RAND_THRESHOLD = 0.5

    def __init__(
        self,
        key: jax.Array,
        batch_size: int = 16,
        num_prototypes: int = 10,
        dim_prototypes: int = 100,
        num_data: int = 1000,
        p_noise: float = 0.3,
    ):
        assert 0 <= p_noise < self.RAND_THRESHOLD, f"Invalid {p_noise=}"
        assert num_data >= num_prototypes, f"Invalid {num_data=}, {num_prototypes=}"
        assert batch_size > 1, f"Invalid {batch_size=}"
        self.num_prototypes = int(num_prototypes)
        self.dim_prototypes = int(dim_prototypes)
        self.num_data = int(num_data)
        self.p_noise = float(p_noise)
        self.batch_size = int(batch_size)
        self.num_batches = -(-self.num_data // self.batch_size)  # ceil division

        key_prototypes, key_data = jax.random.split(key)
        self._create_prototypes(key_prototypes)
        self._create_data(key_data)

    def __iter__(self):
        """Yield batches `(x, y)` as in the original implementation."""
        return zip(
            jnp.array_split(self.x, self.num_batches),
            jnp.array_split(self.y, self.num_batches),
            strict=True,
        )

    def _create_prototypes(self, key: jax.Array) -> None:
        """Generate ±1 prototypes (float32)."""
        # Use rademacher → {−1,+1} with explicit float32 dtype.
        self.prototypes = jax.random.rademacher(
            key, shape=(self.num_prototypes, self.dim_prototypes), dtype=jnp.float32
        )

    def _create_data(self, key: jax.Array) -> None:
        """Generate dataset by repeating prototypes and flipping signs with prob. `p_noise`."""
        # Build OVA labels: +1 on diag, −1 elsewhere, then repeat to length N.
        self.y = jnp.full(
            shape=(self.num_prototypes, self.num_prototypes), fill_value=-1.0, dtype=jnp.float32
        )
        self.y = self.y.at[jnp.diag_indices_from(self.y)].set(1.0)
        self.y = jnp.repeat(
            self.y,
            self.num_data // self.num_prototypes + 1,
            axis=0,
            total_repeat_length=self.num_data,
        )

        # Repeat prototypes to length N, then flip signs with probability p_noise.
        key, carry = jax.random.split(key)
        self.x = jnp.repeat(
            self.prototypes,
            self.num_data // self.num_prototypes + 1,
            axis=0,
            total_repeat_length=self.num_data,
        )
        flip_mask = jax.random.bernoulli(carry, p=1 - self.p_noise, shape=self.x.shape) * 2.0 - 1.0
        self.x = self.x * flip_mask
        # Shuffle x and y in sync.
        shuffle = jax.random.permutation(key, self.num_data)
        self.x = self.x[shuffle].astype(jnp.float32)
        self.y = self.y[shuffle].astype(jnp.float32)
```

## Build Model & State

Here we define a simple model with one hidden layer and fully-connected adapters,
similar to a perceptron with one hidden layer, except that the hidden layer has internal
recurrency.

The topology of the layer map is the following:
- Layer 1 (hidden) receives from input (0, forth), itself (1, recurrent), and labels (2, back).
- Layer 2 (output) receives from hidden (1) and itself (2).

Some first comments:

- Notice that we do not define a input layer, that would correspond to layer 0 in the receivers. This is totally fine and intended.
The input in this case is simply a sent message and never updated. If we dont define layer 0 in the receivers, the first component of the
state is fixed.
- You should inspect the implementation of the OutputLayer. It does not have an internal state, parameters, and the `__call__` function
returns an array of zeros. It is basically a sink that aggregates messages from all layers that contribute to the output and sums them.
This behaviour can change in the future with the definition of new OutputLayers with a more complex logic, but for now it is basically an
aggregator.

```python
DIM_DATA = 100
NUM_DATA = 1000
NUM_LABELS = 10
DIM_HIDDEN = 256
THRESHOLD_OUT = 3.5
THRESHOLD_IN = 3.5
THRESHOLD_J = 0.5
STRENGTH_BACK = 0.3
STRENGTH_FORTH = 1.0
J_D = 0.5

# Global state with three buffers: input (0), hidden (1), output/labels (2)
state = SequentialState((DIM_DATA, DIM_HIDDEN, NUM_LABELS))

# Distinct keys per module to avoid accidental correlations.
master_key = jax.random.key(seed=44)
keys = jax.random.split(master_key, num=5)

layer_map = {
    # Hidden row (1): from input (0), self (1), and labels (2)
    1: {
        0: FullyConnected(
            in_features=DIM_DATA,
            out_features=DIM_HIDDEN,
            strength=STRENGTH_FORTH,
            threshold=THRESHOLD_IN,
            key=keys[0],
        ),
        1: RecurrentDiscrete(features=DIM_HIDDEN, j_d=J_D, threshold=THRESHOLD_J, key=keys[1]),
        2: FrozenFullyConnected(
            in_features=NUM_LABELS,
            out_features=DIM_HIDDEN,
            strength=STRENGTH_BACK,
            threshold=0.0,
            key=keys[2],
        ),
    },
    # Output row (2): from hidden (1), and itself (2)
    2: {
        1: FullyConnected(
            in_features=DIM_HIDDEN,
            out_features=NUM_LABELS,
            strength=1.0,
            threshold=THRESHOLD_OUT,
            key=keys[3],
        ),
        2: OutputLayer(),  # the 2-2 __call__ is a vector of zeros, it does not contribute
    },
}
layer_map = LayerMap.from_dict(layer_map)

# Trainable orchestrator built from the fixed topology.
orchestrator = SequentialOrchestrator(layers=layer_map)
```

## Optimizer

We choose Adam with `lr=5e-3`.
As you can see we can call `eqx.filter` directly on the orchestrator, since it is a pytree.
We also show how to update the model with the function `update(orchestrator, state, optimizer)`.

```python
optimizer = optax.adam(5e-3)
opt_state = optimizer.init(eqx.filter(orchestrator, eqx.is_inexact_array))


def update(
    orchestrator: SequentialOrchestrator, state: SequentialState, optimizer, optimizer_state, rng
) -> tuple[SequentialOrchestrator, Any]:
    """Compute and applies the updates and returns the updated model.

    It also returns the optimizer state, typed as Any for now.
    """
    # 1) Local deltas (orchestrator-shaped deltas).
    grads = orchestrator.backward(state, rng=rng)

    # 2) Optax over the orchestrator (tree structures match by construction).
    # This is common equinox + optax pattern, used in the same way when training
    # "regular" deep learning models.
    # First we filter fields with equinox and then we give them to the optimizer.
    # This allows us to handle complex pytrees with static fields seamlessly during
    # training.
    params = eqx.filter(orchestrator, eqx.is_inexact_array)
    grads = eqx.filter(grads, eqx.is_inexact_array)

    # 3) We compute the updates and apply them to our model
    updates, opt_state = optimizer.update(grads, optimizer_state, params=params)
    orchestrator = eqx.apply_updates(orchestrator, updates)

    # 4) We return the updated model and the optimizer state
    return orchestrator, opt_state
```


## Dynamics helpers

We now define two simple functions that run during training, they are extremely simple.
During run_dynamics_training we have two phases: a first one where we compute all messages and run the dynamics
with both forward and backward messages. We run this phase for a fixed number of steps.
Then, we do a second phase where we suppress all messages "going backward", we run this dynamics for a fixed
number of steps and we obtain a second fixed point s^*.

During inference, we only run the dynamics with suppressed messages from the right.

```python


def run_dynamics_training(
    orch: SequentialOrchestrator,
    s,
    rng: jax.Array,
    steps: int,
):
    """Run `steps` iterations using ALL messages (clamped phase).

    Note: Kept identical to your working version for consistency.
    """
    for _ in range(steps):
        s, rng = orch.step(s, rng=rng)
    for _ in range(steps):
        s, rng = orch.step_inference(s, rng=rng)
    return s, rng


def run_dynamics_inference(
    orch: SequentialOrchestrator,
    s,
    rng: jax.Array,
    steps: int,
):
    """Run `steps` iterations discarding rightward/backward messages (free phase)."""
    for _ in range(steps):
        s, rng = orch.step_inference(s, rng=rng)
    return s, rng
```


## Train step

This function summarizes the whole training protocol, for a single batch.

Protocol per batch:
1. Initialize/clamp the global state with `(x, y)`.
2. Run **training dynamics** for `2 * T_train` steps.
3. Update the model with `update`, as seen before.

```python


def train_step(
    orch: SequentialOrchestrator,
    s: SequentialState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    rng: jax.Array,
    *,
    opt_state,
    optimizer,
    t_train: int = 3,
):
    # 1) Clamp current batch (inputs & labels).
    s = s.init(x, y)

    # 2) Training dynamics (kept as-is).
    s, rng = run_dynamics_training(orch, s, rng, steps=t_train)

    # 3) Update the model
    rng, update_key = jax.random.split(rng)
    orch, opt_state = update(orch, s, optimizer, opt_state, update_key)
    return orch, rng, opt_state
```


## Eval step

Initializes with inputs only (labels are just for metrics), then runs the inference dynamics and computes metrics.


```python
def eval_step(
    orch: SequentialOrchestrator,
    s: SequentialState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    rng: jax.Array,
    *,
    t_eval: int = 5,
) -> tuple[SequentialOrchestrator, SequentialState, dict, jax.Array]:
    s = s.init(x, None)
    s, rng = run_dynamics_inference(orch, s, rng, steps=t_eval)
    s, rng = orchestrator.predict(s, rng)
    y_pred = s[-1]
    metrics = {"acc": batch_accuracy(y, y_pred)}
    return metrics, rng
```

## Run the training

Finally, we build a small `PrototypeData` stream, train for a few epochs using `train_step`
per batch, and evaluate on the same stream.

```python
# Constants
P_NOISE = 0.3
BATCH_SIZE = 16
EPOCHS = 3
T_TRAIN = 10  # training dynamics steps per batch
T_EVAL = 10  # short inference steps for monitoring

# RNGs
master_key = jax.random.key(59)
master_key, data_key = jax.random.split(master_key)

# Data
data = PrototypeData(
    key=data_key,
    batch_size=BATCH_SIZE,
    num_prototypes=NUM_LABELS,
    dim_prototypes=DIM_DATA,
    num_data=NUM_DATA,
    p_noise=P_NOISE,
)
print(f"Dataset: x.shape={tuple(data.x.shape)}  y.shape={tuple(data.y.shape)}")

# Training config

history = {"acc": []}

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
    for x_batch, y_batch in data:
        # Keep batch arrays float32 (consistency)
        master_key, train_key, eval_key = jax.random.split(master_key, num=3)
        orchestrator, master_key, opt_state = train_step(
            orchestrator,
            state,
            x_batch,
            y_batch,
            rng=train_key,
            opt_state=opt_state,
            optimizer=optimizer,
            t_train=T_TRAIN,
        )
        metrics, rng = eval_step(orchestrator, state, x_batch, y_batch, eval_key)
    history["acc"].append(metrics["acc"])
    print(f"Epoch {epoch} done in {time.time()-t0:.2f}s")
    print(f"Mean accuracy={float(jnp.mean(jnp.array(history['acc']))):.3f}")
```

## Final evaluation (demo)

Single pass over the same iterator; replace with a held‑out set in practice.

```python
eval_acc = []
for x_b, y_b in data:
    x_batch = x_b.astype(jnp.float32)
    y_batch = y_b.astype(jnp.float32)
    master_key, step_key = jax.random.split(master_key)
    metrics, master_key = eval_step(
        orchestrator,
        state,
        x_batch,
        y_batch,
        rng=step_key,
        t_eval=T_EVAL,
    )
    eval_acc.append(metrics["acc"])

print("\n=== Final evaluation summary ===")
print(f"Accuracy={float(jnp.mean(jnp.array(eval_acc))):.3f}")
```
