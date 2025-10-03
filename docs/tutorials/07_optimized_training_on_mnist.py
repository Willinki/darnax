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
# # 06 — Optimized training on MNIST
#
# As before, this notebook shows an end‑to‑end training loop.
# We will see how to optimize the training loop with just-in-time compilation on GPU:

# %%
# --- Imports ---------------------------------------------------------------
import logging
import time
from collections.abc import Iterator
from typing import Any

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# %% [markdown]
# ## Utilities (metrics & summaries)
#
# Small helpers used for monitoring. Labels are OVA ±1 and predictions are
# decoded via `argmax` over the output scores.


# %%
def batch_accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Accuracy with ±1 OVA labels (class = argmax along last dim)."""
    y_true_idx = jnp.argmax(y_true, axis=-1)
    y_pred_idx = jnp.argmax(y_pred, axis=-1)
    return float(jnp.mean(y_true_idx == y_pred_idx))


# %% [markdown]
# ## Dataset: MNIST with optional linear projection and sign.
# The data is obtained by taking an image from MNIST,


# %%
class MNISTData:
    """MNIST dataset with optional linear projection and sign."""

    TOTAL_SIZE_PER_CLASS = 6000  # train split
    TEST_SIZE_PER_CLASS = 1000  # test split
    NUM_CLASSES = 10
    FLAT_DIM = 28 * 28

    def __init__(
        self,
        key: jax.Array,
        batch_size: int = 16,
        linear_projection: int | None = 100,
        apply_sign_transform: bool = True,
        num_images_per_class: int = TOTAL_SIZE_PER_CLASS,
    ):
        """Create MNIST dataset with optional projection and sign non-linearity.

        Define an object that contains MNIST images, on which we can iterate.
        Images are flattened pixelwise. Optionally, if `linear_projection` is an
        integer, the images are projected with a linear transformation to the
        specified dimension. After projection, if `apply_sign_transform=True`,
        we apply the `sign()` function element-wise, resulting in ±1 values.

        Parameters
        ----------
        key
            RNG key for generation (projection matrix, shuffling, sampling).
        batch_size
            Batch size for the training/eval iterators. Must be > 1.
        linear_projection
            If `int`, project each flattened image with a random linear map to
            this dimensionality. If `None`, no projection is applied.
        apply_sign_transform
            If `True`, apply `jnp.sign` element-wise at the end (after
            projection if any). Zeros are mapped to -1 so outputs are in
            {-1, +1}.
        num_images_per_class
            Total number of images **per class** to include in the **training**
            set (uniform per-class sampling from the 60k train split).
            Cannot be greater than 6000.

        Examples
        --------
        # Entangled-MNIST dataset
        >>> MNISTData(key, batch_size=16, linear_projection=100, apply_sign_transform=True)
        # Regular MNIST
        >>> MNISTData(key, batch_size=16, linear_projection=None, apply_sign_transform=False)

        """
        if not (linear_projection is None or isinstance(linear_projection, int)):
            raise TypeError("`linear_projection` must be `None` or `int`.")
        if batch_size <= 1:
            raise ValueError(f"Invalid batch_size={batch_size!r}; must be > 1.")
        if not (0 < num_images_per_class <= self.TOTAL_SIZE_PER_CLASS):
            raise ValueError(
                f"Invalid num_images_per_class={num_images_per_class!r}; "
                f"must be in [1, {self.TOTAL_SIZE_PER_CLASS}]."
            )

        self.linear_projection = linear_projection
        self.apply_sign_transform = bool(apply_sign_transform)

        # Training set size (eval uses full test set and is independent)
        self.num_data = int(num_images_per_class) * self.NUM_CLASSES
        self.batch_size = int(batch_size)
        self.num_batches = -(-self.num_data // self.batch_size)  # ceil division

        # Materialize datasets (train + eval) and keep a reference returned as well.
        self._create_dataset(key)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def __iter__(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Yield `(x, y)` **training** batches.

        Yields
        ------
        Iterator[Tuple[jax.Array, jax.Array]]
            Batches of inputs and labels.
            - `x`: shape `(batch, D)` where `D` is either 784 or `linear_projection`.
            - `y`: shape `(batch, 10)` with entries in {-1.0, +1.0}; +1.0 at the
              ground-truth class, -1.0 elsewhere.

        """
        return zip(
            jnp.array_split(self.x, self.num_batches),
            jnp.array_split(self.y, self.num_batches),
            strict=True,
        )

    def iter_eval(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Yield `(x_eval, y_eval)` **validation** batches (full test split)."""
        if self.num_eval_batches == 0:
            return iter(())  # empty iterator
        return zip(
            jnp.array_split(self.x_eval, self.num_eval_batches),
            jnp.array_split(self.y_eval, self.num_eval_batches),
            strict=True,
        )

    def __len__(self) -> int:
        """Return number of **training** batches produced by the iterator."""
        return self.num_batches

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _load_mnist_split(split: str) -> tuple[jax.Array, jax.Array]:
        """Load a MNIST split ('train' or 'test'), flattened.

        Parameters
        ----------
        split
            One of {"train", "test"}.

        Returns
        -------
        x : jax.Array
            Array of shape `(N, 784)`, dtype float32 in [0, 1].
        y : jax.Array
            Array of shape `(N,)`, dtype int32 with labels in [0, 9].

        """
        assert split in ["train", "test"]
        mnist = load_dataset("mnist")
        x = jnp.asarray([jnp.array(im) for im in mnist[split]["image"]], dtype=jnp.float32)
        y = jnp.asarray(mnist[split]["label"], dtype=jnp.int16)
        x = x.reshape(x.shape[0], -1) / 255

        return x, y

    @staticmethod
    def _labels_to_pm1(y_scalar: jax.Array, num_classes: int) -> jax.Array:
        """Convert integer labels `(N,)` to ±1 vectors `(N, C)`."""
        one_hot = jax.nn.one_hot(y_scalar, num_classes, dtype=jnp.float32)
        return one_hot * ((num_classes**0.5) / 2 + 0.5) - 0.5

    @staticmethod
    def _random_projection_matrix(
        key: jax.Array,
        out_dim: int,
        in_dim: int,
    ) -> jax.Array:
        """Gaussian random projection matrix with variance scaling.

        Draws `W ~ N(0, 1/in_dim)` so that each output dimension has unit variance
        under i.i.d. unit-variance inputs.
        """
        w = jax.random.normal(key, (out_dim, in_dim), dtype=jnp.float32)
        return w

    @staticmethod
    def _take_per_class(
        key: jax.Array,
        x: jax.Array,
        y: jax.Array,
        k_per_class: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Uniformly sample `k_per_class` examples for each class 0..9."""
        xs: list[jax.Array] = []
        ys: list[jax.Array] = []
        for cls in range(MNISTData.NUM_CLASSES):
            key, sub = jax.random.split(key)
            idx = jnp.where(y == cls)[0]
            if k_per_class > idx.shape[0]:
                raise ValueError(
                    f"Requested {k_per_class} samples for class {cls}, "
                    f"but only {idx.shape[0]} available."
                )
            perm = jax.random.permutation(sub, idx.shape[0])
            take = idx[perm[:k_per_class]]
            xs.append(x[take])
            ys.append(y[take])

        x_out = jnp.concatenate(xs, axis=0)
        y_out = jnp.concatenate(ys, axis=0)
        return x_out, y_out

    def _maybe_project_and_sign(self, w: jax.Array | None, x: jax.Array) -> jax.Array:
        """Apply optional linear projection (via `w`) and optional sign."""
        if w is not None:
            x = (x @ w.T).astype(jnp.float32)
        if self.apply_sign_transform:
            sgn = jnp.sign(x)
            x = jnp.where(sgn == 0, jnp.array(-1, dtype=sgn.dtype), sgn)
        return x

    # --------------------------------------------------------------------- #
    # Dataset materialization (train subset + full test eval)
    # --------------------------------------------------------------------- #
    def _create_dataset(
        self,
        key: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        """Create train/eval datasets and return them.

        Steps
        -----
        1) Load MNIST train split (60k) and test split (10k), flattened.
        2) TRAIN: uniformly sample `num_images_per_class` per class from train.
        3) EVAL: take the full test split (no subsetting).
        4) Apply a shared projection (if enabled) and optional sign to both.
        5) Convert labels to ±1 vectors; shuffle **train** only.

        Returns
        -------
        (x_train, y_train), (x_eval, y_eval)
            Train and eval features/labels. Labels are ±1 vectors.

        """
        key_sample, key_proj, key_shuf_tr = jax.random.split(key, 3)

        # (1) Load splits
        x_tr_all, y_tr_all = self._load_mnist_split("train")
        x_ev_all, y_ev_scalar = self._load_mnist_split("test")

        # (2) TRAIN subset per class
        k_train = self.num_data // self.NUM_CLASSES
        x_tr, y_tr_scalar = self._take_per_class(key_sample, x_tr_all, y_tr_all, k_train)

        # --- NEW: global centering ---
        mu = jnp.mean(x_tr_all, axis=0, dtype=jnp.float32)  # (784,)
        x_tr_all = x_tr_all - mu
        x_ev_all = x_ev_all - mu

        # (3) EVAL full test split (already loaded as x_ev_all, y_ev_scalar)

        # (4) Shared projection/sign
        if self.linear_projection is not None:
            w = self._random_projection_matrix(
                key_proj, int(self.linear_projection), x_tr.shape[-1]
            )
        else:
            w = None
        x_tr = self._maybe_project_and_sign(w, x_tr)
        x_ev = self._maybe_project_and_sign(w, x_ev_all)

        # (5) Labels to ±1; shuffle train only
        y_tr = self._labels_to_pm1(y_tr_scalar, self.NUM_CLASSES)
        perm_tr = jax.random.permutation(key_shuf_tr, x_tr.shape[0])
        x_tr, y_tr = x_tr[perm_tr], y_tr[perm_tr]

        y_ev = self._labels_to_pm1(y_ev_scalar, self.NUM_CLASSES)

        # Store
        self.x, self.y = x_tr, y_tr
        self.x_eval, self.y_eval = x_ev, y_ev

        # Metadata
        self.input_dim = int(self.x.shape[1])
        self.output_dim = self.linear_projection
        self.num_eval_data = int(self.x_eval.shape[0])
        self.num_eval_batches = -(-self.num_eval_data // self.batch_size)  # ceil div

        return (self.x, self.y), (self.x_eval, self.y_eval)

    # --------------------------------------------------------------------- #
    # Nice-to-have representations
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        """Debug-friendly representation."""
        proj = self.linear_projection if self.linear_projection is not None else "None"
        return (
            "MNISTData("
            f"num_data={self.num_data}, batch_size={self.batch_size}, "
            f"linear_projection={proj}, apply_sign_transform={self.apply_sign_transform}, "
            f"input_dim={getattr(self, 'input_dim', 'n/a')}, "
            f"num_eval_data={getattr(self, 'num_eval_data', 'n/a')})"
        )


# %% [markdown]
# ## Build Model & State
#
# The model is exactly the same as the first tutorial. For completeness, we describe it again.
#
# Here we define a simple model with one hidden layer and fully-connected adapters,
# similar to a perceptron with one hidden layer, except that the hidden layer has internal
# recurrency.
#
# The topology of the layer map is the following:
# - Layer 1 (hidden) receives from input (0, forth), itself (1, recurrent), and labels (2, back).
# - Layer 2 (output) receives from hidden (1) and itself (2).
#
# Some first comments:
#
# - Notice that we do not define a input layer, that would correspond to layer 0 in the receivers. This is totally fine and intended.
# The input in this case is simply a sent message and never updated. If we dont define layer 0 in the receivers, the first component of the
# state is fixed.
# - You should inspect the implementation of the OutputLayer. It does not have an internal state, parameters, and the `__call__` function
# returns an array of zeros. It is basically a sink that aggregates messages from all layers that contribute to the output and sums them.
# This behaviour can change in the future with the definition of new OutputLayers with a more complex logic, but for now it is basically an
# aggregator.

# %%
DIM_DATA = 100
NUM_LABELS = MNISTData.NUM_CLASSES
DIM_HIDDEN = 512
THRESHOLD_OUT = 3.0
THRESHOLD_IN = 3.0
THRESHOLD_J = 1.2
STRENGTH_BACK = 1.2
STRENGTH_FORTH = 5.0
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
logger.info("Defined model with orchestrator.")

# %% [markdown]
# ## Optimizer
#
# We choose Adam with `lr=5e-3`.
# As you can see we can call `eqx.filter` directly on the orchestrator, since it is a pytree.
# We also show how to update the model with the function `update(orchestrator, state, optimizer)`.

# %%
# Build a mask: True = trainable, False = frozen (e.g., W_back)
trainable_mask = jax.tree_util.tree_map(
    lambda leaf: isinstance(leaf, jnp.ndarray),  # start from arrays
    eqx.filter(orchestrator, eqx.is_inexact_array),
)
# turn off the mask at the exact leaves belonging to the FrozenFullyConnected W
trainable_mask = eqx.tree_at(lambda o: o.lmap[1][2].W, trainable_mask, False)
optimizer = optax.masked(optax.sgd(0.008), trainable_mask)
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


# %% [markdown]
# ## Dynamics helpers
#
# We now define two simple functions that run during training, they are extremely simple.
# During run_dynamics_training we have two phases: a first one where we compute all messages and run the dynamics
# with both forward and backward messages. We run this phase for a fixed number of steps.
# Then, we do a second phase where we suppress all messages "going backward", we run this dynamics for a fixed
# number of steps and we obtain a second fixed point s^*.
#
# During inference, we only run the dynamics with suppressed messages from the right.

# %%


def run_dynamics_training(
    orch: SequentialOrchestrator,
    s,
    rng: jax.Array,
    steps: int,
):
    """Run `steps` iterations using ALL messages (clamped phase).

    Note: Kept identical to your working version for consistency.
    """
    s, rng = orch.step_inference(s, rng=rng)
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


# %% [markdown]
# ## Train step
#
# This function summarizes the whole training protocol, for a single batch.
#
# Protocol per batch:
# 1. Initialize/clamp the global state with `(x, y)`.
# 2. Run **training dynamics** for `2 * T_train` steps.
# 3. Update the model with `update`, as seen before.

# %%


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
    """Perform a batch update of the model."""
    # 1) Clamp current batch (inputs & labels).
    s = s.init(x, y)

    # 2) Training dynamics (kept as-is).
    s, rng = run_dynamics_training(orch, s, rng, steps=t_train)

    # 3) Update the model
    rng, update_key = jax.random.split(rng)
    orch, opt_state = update(
        orch, s.replace_val(-1, jnp.sign(s[-1])), optimizer, opt_state, update_key
    )
    return orch, rng, opt_state


# %% [markdown]
# ## Eval step
#
# Initializes with inputs only (labels are just for metrics), then runs the inference dynamics and computes metrics.


# %%
def eval_step(
    orch: SequentialOrchestrator,
    s: SequentialState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    rng: jax.Array,
    *,
    t_eval: int = 5,
) -> tuple[SequentialOrchestrator, SequentialState, dict, jax.Array]:
    """Evaluate the model on a single forward."""
    s = s.init(x, None)
    s, rng = run_dynamics_inference(orch, s, rng, steps=t_eval)
    s, rng = orchestrator.predict(s, rng)
    y_pred = s[-1]
    metrics = {"acc": batch_accuracy(y, y_pred)}
    return metrics, rng


# %% [markdown]
# ## Run the training
#
# Finally, we build a small `PrototypeData` stream, train for a few epochs using `train_step`
# per batch, and evaluate on the same stream.

# %%
# Constants
NUM_IMAGES_PER_CLASS = 600
APPLY_SIGN_TRANSFORM = True
BATCH_SIZE = 16
EPOCHS = 5
T_TRAIN = 10  # training dynamics steps per batch
T_EVAL = 10  # short inference steps for monitoring

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
print(f"Dataset: x.shape={tuple(data.x.shape)}  y.shape={tuple(data.y.shape)}")

# Training config

history = {"acc": []}

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
    w_in_norm = jnp.linalg.norm(orchestrator.lmap[1, 0].W)
    w_out_norm = jnp.linalg.norm(orchestrator.lmap[2, 1].W)
    w_back_norm = jnp.linalg.norm(orchestrator.lmap[1, 2].W)
    J_norm = jnp.linalg.norm(orchestrator.lmap[1, 1].J)
    print(f"{w_in_norm=} {w_out_norm=} {J_norm=} {w_back_norm=}")
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
    print(f"Epoch {epoch} done in {time.time() - t0:.2f}s")
    w_in_norm = jnp.linalg.norm(orchestrator.lmap[1, 0].W)
    w_out_norm = jnp.linalg.norm(orchestrator.lmap[2, 1].W)
    w_back_norm = jnp.linalg.norm(orchestrator.lmap[1, 2].W)
    J_norm = jnp.linalg.norm(orchestrator.lmap[1, 1].J)
    print(f"{w_in_norm=} {w_out_norm=} {J_norm=} {w_back_norm=}")

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

    print(f"Accuracy={float(jnp.mean(jnp.array(eval_acc))):.3f}")

# %% [markdown]
# ## Final evaluation (demo)
#
# Single pass over the same iterator; replace with a held‑out set in practice.

# %%
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
