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

# 08 — Training on MNIST with **darnax** using `DynamicalTrainer` (JAX-first, CPU)

We employ the same network/topology as the previous tutorial—but we employ
an object called DynamicalTrainer that encapsulates the training and validation
logic in a pytorch-lightning style. Many trainers are available.

```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jax
import equinox as eqx
import jax.numpy as jnp
import optax
from datasets import load_dataset

from darnax.layer_maps.sparse import LayerMap
from darnax.modules.fully_connected import FrozenFullyConnected, FullyConnected
from darnax.modules.input_output import OutputLayer
from darnax.modules.recurrent import RecurrentDiscrete
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
```

## Accuracy helper (±1 OVA labels)


```python
def batch_accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Compute the accuracy for a given batch."""
    y_true_idx = jnp.argmax(y_true, axis=-1)
    y_pred_idx = jnp.argmax(y_pred, axis=-1)
    return jnp.mean((y_true_idx == y_pred_idx).astype(jnp.float32))
```


## Minimal MNIST data (projection + optional sign)


```python
class MNISTData:
    """Simple implementation of MNIST dataset."""

    NUM_CLASSES = 10
    FLAT_DIM = 28 * 28
    TOTAL_SIZE_PER_CLASS = 5900
    TEST_SIZE_PER_CLASS = 1000

    def __init__(
        self,
        key: jax.Array,
        batch_size: int = 64,
        linear_projection: int | None = 100,
        apply_sign_transform: bool = True,
        num_images_per_class: int = TOTAL_SIZE_PER_CLASS,
    ):
        """Initialize MNIST."""
        if batch_size <= 1:
            raise ValueError("batch_size must be > 1")
        if not (0 < num_images_per_class <= self.TOTAL_SIZE_PER_CLASS):
            raise ValueError("num_images_per_class out of range")
        self.linear_projection = linear_projection
        self.apply_sign_transform = bool(apply_sign_transform)
        self.num_data = int(num_images_per_class) * self.NUM_CLASSES
        self.batch_size = int(batch_size)
        self.num_batches = -(-self.num_data // self.batch_size)
        self._build(key)
        self._train_bounds = [
            (i * self.batch_size, min((i + 1) * self.batch_size, self.num_data))
            for i in range(self.num_batches)
        ]
        self.num_eval_data = int(self.x_eval.shape[0])
        self.num_eval_batches = -(-self.num_eval_data // self.batch_size)
        self._eval_bounds = [
            (i * self.batch_size, min((i + 1) * self.batch_size, self.num_eval_data))
            for i in range(self.num_eval_batches)
        ]

    # public API
    def __iter__(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        for lo, hi in self._train_bounds:
            yield self.x[lo:hi], self.y[lo:hi]

    def iter_eval(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        for lo, hi in self._eval_bounds:
            yield self.x_eval[lo:hi], self.y_eval[lo:hi]

    def __len__(self) -> int:
        return self.num_batches

    # internals
    @staticmethod
    def _load(split: str) -> tuple[jax.Array, jax.Array]:
        ds = load_dataset("mnist")
        x = jnp.asarray([jnp.array(im) for im in ds[split]["image"]], dtype=jnp.float32)
        y = jnp.asarray(ds[split]["label"], dtype=jnp.int32)
        x = x.reshape(x.shape[0], -1) / 255.0
        return x, y

    @staticmethod
    def _labels_to_pm1_scaled(y_scalar: jax.Array, C: int) -> jax.Array:
        one_hot = jax.nn.one_hot(y_scalar, C, dtype=jnp.float32)
        return one_hot * (C**0.5 / 2.0) - 0.5

    @staticmethod
    def _rand_proj(key: jax.Array, out_dim: int, in_dim: int) -> jax.Array:
        return jax.random.normal(key, (out_dim, in_dim), dtype=jnp.float32) / jnp.sqrt(in_dim)

    @staticmethod
    def _take_per_class(key: jax.Array, x: jax.Array, y: jax.Array, k: int):
        xs, ys = [], []
        for cls in range(MNISTData.NUM_CLASSES):
            key, sub = jax.random.split(key)
            idx = jnp.where(y == cls)[0]
            perm = jax.random.permutation(sub, idx.shape[0])
            take = idx[perm[:k]]
            xs.append(x[take])
            ys.append(y[take])
        return jnp.concatenate(xs, 0), jnp.concatenate(ys, 0)

    def _maybe_proj_sign(self, w: jax.Array | None, x: jax.Array) -> jax.Array:
        if w is not None:
            x = (x @ w.T).astype(jnp.float32)
        if self.apply_sign_transform:
            sgn = jnp.sign(x)
            x = jnp.where(sgn == 0, jnp.array(-1.0, dtype=sgn.dtype), sgn)
        return x

    def _build(self, key: jax.Array) -> None:
        key_sample, key_proj, key_shuffle = jax.random.split(key, 3)
        x_tr_all, y_tr_all = self._load("train")
        x_ev_all, y_ev_scalar = self._load("test")
        k_train = self.num_data // self.NUM_CLASSES
        x_tr, y_tr_scalar = self._take_per_class(key_sample, x_tr_all, y_tr_all, k_train)
        w = (
            self._rand_proj(key_proj, int(self.linear_projection), x_tr.shape[-1])
            if self.linear_projection
            else None
        )
        x_tr = self._maybe_proj_sign(w, x_tr)
        x_ev = self._maybe_proj_sign(w, x_ev_all)
        y_tr = self._labels_to_pm1_scaled(y_tr_scalar, self.NUM_CLASSES)
        perm_tr = jax.random.permutation(key_shuffle, x_tr.shape[0])
        self.x, self.y = x_tr[perm_tr], y_tr[perm_tr]
        self.x_eval = x_ev
        self.y_eval = self._labels_to_pm1_scaled(y_ev_scalar, self.NUM_CLASSES)
        self.input_dim = int(self.x.shape[1])
```

## Model topology (LayerMap + Orchestrator)
One hidden layer:
- 0→1 forward (FullyConnected)
- 1→1 recurrent (RecurrentDiscrete)
- 2→1 teacher/clamp (FrozenFullyConnected)
- 1→2 readout (FullyConnected)
- 2→2 aggregator (OutputLayer)

```python
DIM_DATA = 100
NUM_LABELS = MNISTData.NUM_CLASSES
DIM_HIDDEN = 300

THRESHOLD_OUT = 1.0
THRESHOLD_IN = 1.0
THRESHOLD_J = 1.0
STRENGTH_BACK = 0.5
STRENGTH_FORTH = 5.0
J_D = 0.5

state = SequentialState((DIM_DATA, DIM_HIDDEN, NUM_LABELS))

master_key = jax.random.key(44)
keys = jax.random.split(master_key, 5)

layer_map = {
    1: {
        0: FullyConnected(
            in_features=DIM_DATA,
            out_features=DIM_HIDDEN,
            strength=STRENGTH_FORTH,
            threshold=THRESHOLD_IN,
            key=keys[0],
        ),
        1: RecurrentDiscrete(
            features=DIM_HIDDEN,
            j_d=J_D,
            threshold=THRESHOLD_J,
            key=keys[1],
        ),
        2: FrozenFullyConnected(
            in_features=NUM_LABELS,
            out_features=DIM_HIDDEN,
            strength=STRENGTH_BACK,
            threshold=0.0,
            key=keys[2],
        ),
    },
    2: {
        1: FullyConnected(
            in_features=DIM_HIDDEN,
            out_features=NUM_LABELS,
            strength=1.0,
            threshold=THRESHOLD_OUT,
            key=keys[3],
        ),
        2: OutputLayer(),
    },
}
layer_map = LayerMap.from_dict(layer_map)
orchestrator = SequentialOrchestrator(layers=layer_map)
logger.info("Model initialized with SequentialOrchestrator.")
```

## Optimizer & `DynamicalTrainer`
The trainer owns Optax state and RNG and exposes `fit_epoch` / `eval_epoch`.

```python
# Experiment knobs
NUM_IMAGES_PER_CLASS = 5400
APPLY_SIGN_TRANSFORM = True
BATCH_SIZE = 16
EPOCHS = 5
T_WARMUP = 1  # warmup phase
T_FREE = 7  # free phase iterations
T_CLAMPED = 7  # clamped phase iterations
T_EVAL = 14  # validation phase

# RNGs
run_key = jax.random.key(59)
run_key, data_key = jax.random.split(run_key)

# Data
data = MNISTData(
    key=data_key,
    batch_size=BATCH_SIZE,
    linear_projection=DIM_DATA,
    apply_sign_transform=APPLY_SIGN_TRANSFORM,
    num_images_per_class=NUM_IMAGES_PER_CLASS,
)
print(f"Dataset — x.shape={tuple(data.x.shape)}, y.shape={tuple(data.y.shape)}")

# Optimizer
optimizer = optax.adam(2e-3)
optimizer_state = optimizer.init(eqx.filter(orchestrator, eqx.is_inexact_array))

# Trainer
trainer = DynamicalTrainer(
    orchestrator=orchestrator,
    state=state,
    optimizer=optimizer,
    optimizer_state=optimizer_state,
    warmup_n_iter=T_WARMUP,
    train_clamped_n_iter=T_CLAMPED,
    train_free_n_iter=T_FREE,
    eval_n_iter=T_EVAL,
)
```

## Fit (compact epoch loop)

```python
logs = []
run_key, rng = jax.random.split(run_key)
logger.info("Starting training...")
for epoch in range(1, EPOCHS + 1):
    logger.info(f"{epoch=}")
    for x, y in data:
        rng = trainer.train_step(x, y, rng)
    eval_accuracies = []
    for x, y in data.iter_eval():
        rng, metrics = trainer.eval_step(x, y, rng)
        eval_accuracies.append(metrics["accuracy"])
    print(f"Eval accuracy: {sum(eval_accuracies) / len(eval_accuracies)}")
```

## Final evaluation

```python
print("\n=== Final evaluation ===")
for x, y in data.iter_eval():
    rng, metrics = trainer.eval_step(x, y, rng)
    eval_accuracies.append(metrics["accuracy"])
print(f"Eval accuracy: {sum(eval_accuracies) / len(eval_accuracies)}")
```

## Recap
You kept the **LayerMap/State/Orchestrator** design and local-plasticity updates,
but moved outer-loop mechanics into a thin, reusable `DynamicalTrainer`:
a stable API (`train_step`, `eval_step`).
