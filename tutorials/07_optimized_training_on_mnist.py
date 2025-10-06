# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # 07 — Training on MNIST with **darnax** (JAX-first, CPU)
#
# This tutorial is for readers who want to **understand how darnax is used** in practice:
# how a network is assembled from modules, how the **orchestrator** runs recurrent dynamics,
# and how **local plasticity** integrates with Equinox/Optax to update parameters *without*
# backpropagation.
#
# We’ll implement a compact, JAX-friendly training loop:
#
# - **Only the outer steps are `jit`-compiled**: `train_step`, `eval_step`.
# - Recurrent dynamics use **`jax.lax.scan`** (no Python loops inside compiled code).
# - The dataset iterator is **slice-based** (avoid `array_split`), better for accelerators and CPUs.
# - We **stay on CPU** to keep the focus on design; the code is accelerator-ready.

# %%
# --- Imports ---------------------------------------------------------------
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from datasets import load_dataset

from darnax.layer_maps.sparse import LayerMap
from darnax.modules.fully_connected import FrozenFullyConnected, FullyConnected
from darnax.modules.input_output import OutputLayer
from darnax.modules.recurrent import RecurrentDiscrete
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# %% [markdown]
# ## 1) What darnax gives you
#
# darnax decomposes a model into **modules** connected by a **LayerMap**. At runtime,
# an **orchestrator** applies a **message-passing schedule** over a **state** vector.
#
# - **Modules** (edges/diagonals) consume a sender buffer and emit a message to a receiver.
#   Some modules are trainable; some are frozen; some are “diagonal” (operate on a buffer itself).
# - The **LayerMap** defines the fixed topology: for each receiver row `i`, which senders `j`
#   contribute, and with which module.
# - The **SequentialOrchestrator** drives the update order (left→right, recurrent self, right→left
#   as needed) and exposes:
#   - `step`: full dynamics (all messages allowed).
#   - `step_inference`: inference dynamics (typically suppress “backward” messages).
#   - `backward`: compute **local** parameter deltas from the current state (no backprop).
#   - `predict`: produce output scores in the final buffer.
# - The **State** is a fixed-shape tuple of buffers `(input, hidden, output)`. You **clamp**
#   inputs (and possibly labels) by writing them into the state, then run dynamics to a fixed point.

# %% [markdown]
# ## 2) A tiny metric helper
#
# Labels are One-Vs-All (OVA) in ±1. We decode predictions via `argmax`.
#

# %%
def batch_accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Accuracy with ±1 OVA labels (class = argmax along last dim)."""
    y_true_idx = jnp.argmax(y_true, axis=-1)
    y_pred_idx = jnp.argmax(y_pred, axis=-1)
    return jnp.mean((y_true_idx == y_pred_idx).astype(jnp.float32))


# %% [markdown]
# ## 3) Dataset object designed for JAX
#
# We keep the data pipeline deliberately simple to highlight the model mechanics:
#
# - **Materialize once**: training subset and full test split live as device arrays.
# - **Shared projection**: an optional linear projection (same matrix for train/test) reduces
#   dimensionality and can be followed by a `sign` transform (Entangled-MNIST style).
# - **Slice-based batches**: iterators yield contiguous chunks; no list materialization or
#   `array_split`.
# - **Label scaling** (your original choice):
#   - true class → `+√C / 2`
#   - others → `−0.5`
#
# This scaling biases the output field to favor the target class during clamped dynamics.
#

# %%
class MNISTData:
    """MNIST dataset with optional linear projection and sign; slice-based iterators.

    Design:
      - Single in-memory materialization (train subset, full test).
      - Shared projection across splits.
      - Deterministic batch slicing (precomputed ranges).
    """

    TOTAL_SIZE_PER_CLASS = 5900  # train split
    TEST_SIZE_PER_CLASS = 1000  # test split
    NUM_CLASSES = 10
    FLAT_DIM = 28 * 28

    def __init__(
        self,
        key: jax.Array,
        batch_size: int = 64,
        linear_projection: int | None = 100,
        apply_sign_transform: bool = True,
        num_images_per_class: int = TOTAL_SIZE_PER_CLASS,
    ):
        """Initialize the dataset object."""
        # Lightweight validation; fail fast on easy mistakes.
        if not (linear_projection is None or isinstance(linear_projection, int)):
            raise TypeError("`linear_projection` must be `None` or `int`.")
        if batch_size <= 1:
            raise ValueError(f"Invalid batch_size={batch_size!r}; must be > 1.")
        if not (0 < num_images_per_class <= self.TOTAL_SIZE_PER_CLASS):
            raise ValueError(f"`num_images_per_class` must be in [1, {self.TOTAL_SIZE_PER_CLASS}]")

        self.linear_projection = linear_projection
        self.apply_sign_transform = bool(apply_sign_transform)

        self.num_data = int(num_images_per_class) * self.NUM_CLASSES
        self.batch_size = int(batch_size)
        self.num_batches = -(-self.num_data // self.batch_size)  # ceil div

        # Build arrays once.
        self._create_dataset(key)

        # Precompute slicing ranges for train/eval.
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

    # ------------------------------- Public API ------------------------------- #
    def __iter__(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Yield `(x, y)` training batches by contiguous slicing."""
        for lo, hi in self._train_bounds:
            yield self.x[lo:hi], self.y[lo:hi]

    def iter_eval(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Yield `(x_eval, y_eval)` validation batches (full test split)."""
        for lo, hi in self._eval_bounds:
            yield self.x_eval[lo:hi], self.y_eval[lo:hi]

    def __len__(self) -> int:
        """Return the number of batches."""
        return self.num_batches

    # ------------------------------ Internals ------------------------------ #
    @staticmethod
    def _load_mnist_split(split: str) -> tuple[jax.Array, jax.Array]:
        """Load MNIST split and flatten to (N, 784)."""
        assert split in ["train", "test"]
        ds = load_dataset("mnist")
        x = jnp.asarray([jnp.array(im) for im in ds[split]["image"]], dtype=jnp.float32)
        y = jnp.asarray(ds[split]["label"], dtype=jnp.int32)
        x = x.reshape(x.shape[0], -1) / 255.0
        return x, y

    @staticmethod
    def _labels_to_pm1_scaled(y_scalar: jax.Array, num_classes: int) -> jax.Array:
        """Original scaling: +√C/2 at the true class, −0.5 elsewhere."""
        one_hot = jax.nn.one_hot(y_scalar, num_classes, dtype=jnp.float32)
        return one_hot * (num_classes**0.5 / 2.0) - 0.5

    @staticmethod
    def _random_projection_matrix(key: jax.Array, out_dim: int, in_dim: int) -> jax.Array:
        """Gaussian projection with variance 1/in_dim to keep outputs ~unit variance."""
        return jax.random.normal(key, (out_dim, in_dim), dtype=jnp.float32) / jnp.sqrt(in_dim)

    @staticmethod
    def _take_per_class(
        key: jax.Array, x: jax.Array, y: jax.Array, k_per_class: int
    ) -> tuple[jax.Array, jax.Array]:
        """Uniformly sample `k_per_class` examples for each class 0..9."""
        xs, ys = [], []
        for cls in range(MNISTData.NUM_CLASSES):
            key, sub = jax.random.split(key)
            idx = jnp.where(y == cls)[0]
            if k_per_class > idx.shape[0]:
                raise ValueError(
                    f"Requested {k_per_class} for class {cls}, but only {idx.shape[0]} available."
                )
            perm = jax.random.permutation(sub, idx.shape[0])
            take = idx[perm[:k_per_class]]
            xs.append(x[take])
            ys.append(y[take])
        return jnp.concatenate(xs, axis=0), jnp.concatenate(ys, axis=0)

    def _maybe_project_and_sign(self, w: jax.Array | None, x: jax.Array) -> jax.Array:
        """Apply optional linear projection + optional sign nonlinearity (Entangled-MNIST)."""
        if w is not None:
            x = (x @ w.T).astype(jnp.float32)
        if self.apply_sign_transform:
            sgn = jnp.sign(x)
            x = jnp.where(sgn == 0, jnp.array(-1.0, dtype=sgn.dtype), sgn)
        return x

    def _create_dataset(self, key: jax.Array) -> None:
        """Materialize train subset and full test split with consistent preprocessing."""
        key_sample, key_proj, key_shuf_tr = jax.random.split(key, 3)

        # Load raw splits.
        x_tr_all, y_tr_all = self._load_mnist_split("train")
        x_ev_all, y_ev_scalar = self._load_mnist_split("test")

        # Uniform per-class sampling.
        k_train = self.num_data // self.NUM_CLASSES
        x_tr, y_tr_scalar = self._take_per_class(key_sample, x_tr_all, y_tr_all, k_train)

        # Shared projection/sign across splits.
        w = (
            self._random_projection_matrix(key_proj, int(self.linear_projection), x_tr.shape[-1])
            if self.linear_projection is not None
            else None
        )
        x_tr = self._maybe_project_and_sign(w, x_tr)
        x_ev = self._maybe_project_and_sign(w, x_ev_all)

        # Labels with original scaling; shuffle train only.
        y_tr = self._labels_to_pm1_scaled(y_tr_scalar, self.NUM_CLASSES)
        perm_tr = jax.random.permutation(key_shuf_tr, x_tr.shape[0])
        self.x, self.y = x_tr[perm_tr], y_tr[perm_tr]

        self.x_eval = x_ev
        self.y_eval = self._labels_to_pm1_scaled(y_ev_scalar, self.NUM_CLASSES)

        # Convenience metadata.
        self.input_dim = int(self.x.shape[1])


# %% [markdown]
# ## 4) Model topology as a LayerMap
#
# We build a minimal network with **one hidden layer** and an **output sink**:
#
# - Receiver row **1 (hidden)** gets messages from:
#   - **0 (input)** via `FullyConnected` (forward path)
#   - **1 (itself)** via `RecurrentDiscrete` (internal recurrency)
#   - **2 (labels)** via `FrozenFullyConnected` (backward/clamping path)
# - Receiver row **2 (output)** gets:
#   - **1 (hidden)** via `FullyConnected` (readout)
#   - **2 (itself)** via `OutputLayer` (diagonal sink/aggregator, returns zeros)
#
# The **SequentialState** is `(input, hidden, output)` with fixed sizes.
# The **SequentialOrchestrator** knows how to:
# - aggregate edge messages for each receiver,
# - apply diagonal modules,
# - and run the chosen schedule (`step`, `step_inference`, `predict`, `backward`).

# %%
DIM_DATA = 100
NUM_LABELS = MNISTData.NUM_CLASSES
DIM_HIDDEN = 300

THRESHOLD_OUT = 1.0
THRESHOLD_IN = 1.0
THRESHOLD_J = 1.0
STRENGTH_BACK = 0.5
STRENGTH_FORTH = 5.0
J_D = 0.5

# Global state with three buffers: input (0), hidden (1), output/labels (2)
state = SequentialState((DIM_DATA, DIM_HIDDEN, NUM_LABELS))

# Independent keys for each module (avoid accidental correlations).
master_key = jax.random.key(seed=44)
keys = jax.random.split(master_key, num=5)

layer_map = {
    1: {  # Hidden row receives from input, itself, and labels
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
        2: FrozenFullyConnected(  # clamping/teaching signal, not trainable
            in_features=NUM_LABELS,
            out_features=DIM_HIDDEN,
            strength=STRENGTH_BACK,
            threshold=0.0,
            key=keys[2],
        ),
    },
    2: {  # Output row receives from hidden and aggregates
        1: FullyConnected(
            in_features=DIM_HIDDEN,
            out_features=NUM_LABELS,
            strength=1.0,
            threshold=THRESHOLD_OUT,
            key=keys[3],
        ),
        2: OutputLayer(),  # diagonal sink: produces zeros; acts as aggregator anchor
    },
}
layer_map = LayerMap.from_dict(layer_map)

# Trainable orchestrator built from the fixed topology.
orchestrator = SequentialOrchestrator(layers=layer_map)
logger.info("Model initialized with SequentialOrchestrator.")

# %% [markdown]
# ## 5) Optimizer and the “no-backprop” update
#
# darnax does **not** use backpropagation here. Instead:
#
# 1. Run recurrent dynamics with the current batch **clamped** (inputs + labels in the state).
# 2. Call `orchestrator.backward(state, rng)` to get **local** deltas for every trainable module.
# 3. Apply those deltas using **Optax**—this gives you the familiar optimizer ergonomics.
#
# Notes for JAX compilation:
# - We pass the **optimizer object** as an argument to the jitted functions.
#   Under `eqx.filter_jit`, non-array args are **static**. Reusing the same instance prevents retracing.
# - Only the **optimizer state** (arrays) flows through the jitted code.

# %%
optimizer = optax.adam(2e-3)
opt_state = optimizer.init(eqx.filter(orchestrator, eqx.is_inexact_array))


def _apply_update(
    orch: SequentialOrchestrator,
    s: SequentialState,
    opt_state,
    rng: jax.Array,
    optimizer,
):
    """Compute local deltas via .backward, then apply Optax updates.

    Why separate this helper?
      - Clear separation of concerns (dynamics vs parameter updates).
      - Easier to unit-test and profile independently.
    """
    grads = orch.backward(s, rng=rng)  # local deltas, tree-shaped like `orch`
    params = eqx.filter(orch, eqx.is_inexact_array)  # trainable leaves
    grads = eqx.filter(grads, eqx.is_inexact_array)  # drop non-arrays from grads

    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    orch = eqx.apply_updates(orch, updates)
    return orch, opt_state


# %% [markdown]
# ## 6) Dynamics with `lax.scan` (not jitted directly)
#
# The orchestrator exposes **one-step** transitions:
# - `step(s, rng)` → (s’, rng’): full dynamics (includes backward/label messages).
# - `step_inference(s, rng)` → (s’, rng’): inference-only dynamics (suppress backward messages).
#
# We wrap those into **scans**. These helpers are not jitted on their own; they are traced as
# part of the outer jitted steps. That keeps the code modular and the compiled graph clean.
#

# %%
def _scan_steps(fn, s: SequentialState, rng: jax.Array, steps: int):
    """Scan `steps` times a (s, rng)->(s, rng) transition."""

    def body(carry, _):
        s, rng = carry
        s, rng = fn(s, rng=rng)
        return (s, rng), None

    (s, rng), _ = jax.lax.scan(body, (s, rng), xs=None, length=steps)
    return s, rng


def run_dynamics_training(
    orch: SequentialOrchestrator,
    s: SequentialState,
    rng: jax.Array,
    steps: int,
):
    """Clamped phase (full dynamics) followed by a short free relaxation (inference)."""
    s, rng = _scan_steps(orch.step, s, rng, steps)  # clamped
    s, rng = _scan_steps(orch.step_inference, s, rng, steps)  # free
    return s, rng


def run_dynamics_inference(
    orch: SequentialOrchestrator,
    s: SequentialState,
    rng: jax.Array,
    steps: int,
):
    """Inference-only relaxation to a fixed point."""
    s, rng = _scan_steps(orch.step_inference, s, rng, 2 * steps)
    return s, rng


# %% [markdown]
# ## 7) Outer steps (the only `jit`-compiled functions)
#
# We jit **only** the functions that are called many times and represent the outer boundary
# of our computation:
#
# - **`train_step`** (per batch):
#   1) write `(x, y)` into the state (clamp),
#   2) run clamped + free dynamics,
#   3) compute local deltas and apply the Optax update.
#
# - **`eval_step`** (per batch):
#   1) write `x` only,
#   2) run free dynamics,
#   3) `predict` and compute accuracy.
#
# JIT boundary discipline:
# - **Static args** (optimizer object, Python ints like `t_train`) trigger retraces **only if** they
#   change. Keep them fixed during a run.
#

# %%
@eqx.filter_jit
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
    """Perform a train step in a single batch."""
    # 1) Clamp inputs + labels into the global state.
    s = s.init(x, y)

    # 2) Recurrent dynamics: clamped phase then free relaxation.
    s, rng = run_dynamics_training(orch, s, rng, steps=t_train)

    # 3) Local deltas + Optax update.
    rng, update_key = jax.random.split(rng)
    orch, opt_state = _apply_update(orch, s, opt_state, update_key, optimizer)
    return orch, rng, opt_state


@eqx.filter_jit
def eval_step(
    orch: SequentialOrchestrator,
    s: SequentialState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    rng: jax.Array,
    *,
    t_eval: int = 5,
) -> tuple[float, jax.Array]:
    """Perform a validation step on a single batch."""
    # 1) Clamp inputs only (labels aren't used by dynamics here).
    s = s.init(x, None)

    # 2) Free relaxation to a fixed point.
    s, rng = run_dynamics_inference(orch, s, rng, steps=t_eval)

    # 3) Predict scores from the settled state and measure accuracy.
    s, rng = orch.predict(s, rng)
    y_pred = s[-1]
    acc = batch_accuracy(y, y_pred)
    return acc, rng


# %% [markdown]
# ## 8) Training loop (CPU)
#
# The Python epoch loop shepherds data and RNG. All heavy lifting happens inside the two
# jitted steps above. Practical guidance:
#
# - Keep **array shapes/dtypes** and the **pytrees’ structures** stable across calls.
# - Reuse the **same optimizer instance**; pass its **state** through the jitted code.
# - If you change `t_train`/`t_eval` between calls, expect a retrace (they are static).

# %%
# Experiment knobs
NUM_IMAGES_PER_CLASS = 5400
APPLY_SIGN_TRANSFORM = True
BATCH_SIZE = 16
EPOCHS = 5
T_TRAIN = 10  # clamped + free steps per batch
T_EVAL = 10  # inference steps multiplier (2*T_EVAL iterations)

# RNGs
master_key = jax.random.key(59)
master_key, data_key = jax.random.split(master_key)

# Data
data = MNISTData(
    key=data_key,
    batch_size=BATCH_SIZE,
    linear_projection=DIM_DATA,
    apply_sign_transform=APPLY_SIGN_TRANSFORM,
    num_images_per_class=NUM_IMAGES_PER_CLASS,
)
print(f"Dataset ready — x.shape={tuple(data.x.shape)}, y.shape={tuple(data.y.shape)}")

# Train & evaluate
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    print(f"\n=== Epoch {epoch}/{EPOCHS} ===")

    # Training epoch
    for x_batch, y_batch in data:
        master_key, step_key = jax.random.split(master_key)
        orchestrator, master_key, opt_state = train_step(
            orchestrator,
            state,
            x_batch,
            y_batch,
            rng=step_key,
            opt_state=opt_state,
            optimizer=optimizer,  # static in the JIT sense; same instance every call
            t_train=T_TRAIN,
        )

    # Evaluation epoch (full test split)
    accs = []
    for x_b, y_b in data.iter_eval():
        master_key, step_key = jax.random.split(master_key)
        acc, master_key = eval_step(
            orchestrator,
            state,
            x_b.astype(jnp.float32),
            y_b.astype(jnp.float32),
            rng=step_key,
            t_eval=T_EVAL,
        )
        accs.append(acc)

    acc_epoch = float(jnp.mean(jnp.array(accs)))
    print(f"Eval Accuracy = {acc_epoch:.3f}  |  epoch time: {time.time() - t0:.2f}s")

# %% [markdown]
# ## 9) One-line final report
#
# This is just to have a single scalar you can grep from logs or compare across runs.

# %%
final_accs = []
for x_b, y_b in data.iter_eval():
    master_key, step_key = jax.random.split(master_key)
    acc, master_key = eval_step(
        orchestrator,
        state,
        x_b.astype(jnp.float32),
        y_b.astype(jnp.float32),
        rng=step_key,
        t_eval=T_EVAL,
    )
    final_accs.append(acc)

print("\n=== Final evaluation summary ===")
print(f"Accuracy = {float(jnp.mean(jnp.array(final_accs))):.3f}")

# %% [markdown]
# ## 10) Recap & next steps
#
# You just trained a **recurrent, locally-plastic** network on MNIST using darnax:
#
# - You **declared** topology with a `LayerMap`, not a layer stack.
# - A **state** of fixed buffers `(input, hidden, output)` was **clamped** and then
#   relaxed to a fixed point by the **orchestrator**.
# - You updated parameters using **local deltas** (`orchestrator.backward`) funneled through Optax.
# - You **JIT-compiled the outer loop only**, using `lax.scan` for inner dynamics.
#
# If you’re serious about scaling this:
#
# - **Parallel orchestrators**: swap `SequentialOrchestrator` for a parallel flavor when your
#   graphs grow (careful with data dependencies).
# - **Topology as data**: generate `LayerMap` programmatically (e.g., blocks, conv-like bands).
# - **Per-block scalings**: match initialization and LR magnitudes to each path’s fan-in/out.
# - **Profiling**: dump HLO for `train_step`/`eval_step`, sanity-check fusion and shape stability.
#
# Don’t just accept the defaults—pressure-test the schedule and the rules. If a path doesn’t
# pull its weight (e.g., backward clamp too weak/strong), instrument it and fix it.
