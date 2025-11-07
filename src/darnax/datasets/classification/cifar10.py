from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
from datasets import load_dataset  # type: ignore[import-untyped]

from darnax.datasets.classification.interface import ClassificationDataset

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class CIFAR10(ClassificationDataset):
    """CIFAR-10 dataset with configurable preprocessing.

    Parameters.
    ----------
    batch_size : int, default=64
        Batch size for iterators.
    linear_projection : int or None, default=100
        Output dimension for random projection. If None, uses full 3072 dimensions.
    num_images_per_class : int or None, default=None
        Maximum training images per class. If None, uses full training set.
    label_mode : {"pm1", "ooe", "c-rescale"}, default="c-rescale"
        Label encoding: "pm1" (±1), "ooe" (one-hot), "c-rescale" (scaled).
    x_transform : {"sign", "tanh", "identity"}, default="sign"
        Input transform: "sign" (±1), "tanh", "identity" (no transform).
    validation_fraction : float, default=0.0
        Fraction of training data for validation (0.0 to 1.0).

    References
    ----------
        - https://www.cs.toronto.edu/~kriz/cifar.html
        - https://huggingface.co/datasets/cifar10

    """

    NUM_CLASSES = 10
    FLAT_DIM = 32 * 32 * 3  # CIFAR-10 images are 32x32 RGB

    def __init__(
        self,
        batch_size: int = 64,
        linear_projection: int | None = 100,
        num_images_per_class: int | None = None,
        label_mode: Literal["pm1", "ooe", "c-rescale"] = "c-rescale",
        x_transform: Literal["sign", "tanh", "identity"] = "sign",
        validation_fraction: float = 0.0,
    ) -> None:
        """Initialize CIFAR-10 dataset configuration."""
        if not (linear_projection is None or isinstance(linear_projection, int)):
            raise TypeError("`linear_projection` must be `None` or `int`.")
        if batch_size <= 1:
            raise ValueError(f"Invalid batch_size={batch_size!r}; must be > 1.")
        if num_images_per_class is not None and num_images_per_class <= 0:
            raise ValueError("`num_images_per_class` must be positive or None.")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("`validation_fraction` must be in [0.0, 1.0).")

        self.batch_size = int(batch_size)
        self.linear_projection = linear_projection
        self.num_images_per_class = num_images_per_class
        self.label_mode = label_mode
        self.x_transform = x_transform
        self.validation_fraction = validation_fraction

        self.input_dim: int | None = None
        self.num_classes: int = self.NUM_CLASSES
        self.x_train: jax.Array | None = None
        self.y_train: jax.Array | None = None
        self.x_valid: jax.Array | None = None
        self.y_valid: jax.Array | None = None
        self.x_test: jax.Array | None = None
        self.y_test: jax.Array | None = None
        self._train_bounds: list[tuple[int, int]] = []
        self._valid_bounds: list[tuple[int, int]] = []
        self._test_bounds: list[tuple[int, int]] = []

    def build(self, key: jax.Array) -> None:
        """Load, preprocess, and prepare CIFAR-10 splits."""
        key_sample, key_proj, key_split, key_shuf = jax.random.split(key, 4)

        x_tr_all, y_tr_all = self._load_split("train")
        x_te_all, y_te_all = self._load_split("test")

        if self.num_images_per_class is None:
            x_tr, y_tr = x_tr_all, y_tr_all
        else:
            x_tr, y_tr = self._subsample_per_class(
                key_sample, x_tr_all, y_tr_all, self.num_images_per_class
            )

        if self.validation_fraction > 0.0:
            n_total = x_tr.shape[0]
            n_valid = int(n_total * self.validation_fraction)
            perm = jax.random.permutation(key_split, n_total)
            x_tr, y_tr = x_tr[perm], y_tr[perm]
            x_tr, x_va = x_tr[:-n_valid], x_tr[-n_valid:]
            y_tr, y_va = y_tr[:-n_valid], y_tr[-n_valid:]
        else:
            x_va, y_va = None, None

        w = (
            self._generate_random_projection(key_proj, int(self.linear_projection), self.FLAT_DIM)
            if self.linear_projection is not None
            else None
        )

        x_tr = self._preprocess(w, x_tr)
        x_te = self._preprocess(w, x_te_all)
        if x_va is not None:
            x_va = self._preprocess(w, x_va)

        y_tr = self._encode_labels(y_tr)
        y_te = self._encode_labels(y_te_all)
        y_va = self._encode_labels(y_va) if y_va is not None else None

        perm = jax.random.permutation(key_shuf, x_tr.shape[0])
        self.x_train, self.y_train = x_tr[perm], y_tr[perm]
        self.x_test, self.y_test = x_te, y_te
        if x_va is not None and y_va is not None:
            self.x_valid, self.y_valid = x_va, y_va

        self.input_dim = int(self.x_train.shape[1])

        self._train_bounds = self._compute_bounds(self.x_train.shape[0])
        self._test_bounds = self._compute_bounds(self.x_test.shape[0])
        if self.x_valid is not None:
            self._valid_bounds = self._compute_bounds(self.x_valid.shape[0])

    def __iter__(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Iterate over training batches."""
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Dataset not built. Call `build()` first.")
        for lo, hi in self._train_bounds:
            yield self.x_train[lo:hi], self.y_train[lo:hi]

    def iter_test(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Iterate over test batches."""
        if self.x_test is None or self.y_test is None:
            raise RuntimeError("Dataset not built. Call `build()` first.")
        for lo, hi in self._test_bounds:
            yield self.x_test[lo:hi], self.y_test[lo:hi]

    def iter_valid(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Iterate over validation batches."""
        if self.x_valid is None or self.y_valid is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} has no validation split. "
                "Set validation_fraction > 0 to create one."
            )
        for lo, hi in self._valid_bounds:
            yield self.x_valid[lo:hi], self.y_valid[lo:hi]

    def __len__(self) -> int:
        """Return number of training batches."""
        if not self._train_bounds:
            raise RuntimeError("Dataset not built. Call `build()` first.")
        return len(self._train_bounds)

    def spec(self) -> dict[str, Any]:
        """Return dataset specification."""
        if self.x_train is None or self.y_train is None or self.input_dim is None:
            raise RuntimeError("Dataset not built. Call `build()` first.")

        return {
            "x_shape": (self.input_dim,),
            "x_dtype": self.x_train.dtype,
            "y_shape": (self.NUM_CLASSES,),
            "y_dtype": self.y_train.dtype,
            "num_classes": self.NUM_CLASSES,
            "label_encoding": self.label_mode,
            "projected_dim": self.input_dim if self.linear_projection else None,
        }

    @staticmethod
    def _load_split(split: str) -> tuple[jax.Array, jax.Array]:
        """Load CIFAR10 split from HuggingFace datasets."""
        ds = load_dataset("cifar10", split=split, trust_remote_code=True)
        x_raw = jnp.asarray([jnp.array(im) for im in ds["img"]], dtype=jnp.float32)
        x: jax.Array = (x_raw / 255.0).astype(jnp.float32)
        y: jax.Array = jnp.asarray(ds["label"], dtype=jnp.int32)
        return x, y

    @staticmethod
    def _subsample_per_class(
        key: jax.Array, x: jax.Array, y: jax.Array, k: int
    ) -> tuple[jax.Array, jax.Array]:
        """Sample up to k examples per class."""
        xs, ys = [], []
        for cls in range(CIFAR10.NUM_CLASSES):
            key, sub = jax.random.split(key)
            idx = jnp.where(y == cls)[0]
            n = min(k, int(idx.shape[0]))
            perm = jax.random.permutation(sub, idx.shape[0])
            xs.append(x[idx[perm[:n]]])
            ys.append(y[idx[perm[:n]]])
        return jnp.concatenate(xs), jnp.concatenate(ys)

    @staticmethod
    def _generate_random_projection(key: jax.Array, out_dim: int, in_dim: int) -> jax.Array:
        """Generate random Gaussian projection matrix."""
        return jax.random.normal(key, (out_dim, in_dim), dtype=jnp.float32) / jnp.sqrt(in_dim)

    def _preprocess(self, w: jax.Array | None, x: jax.Array) -> jax.Array:
        """Flatten, project, and transform inputs."""
        x = jnp.reshape(x, (x.shape[0], -1))
        if w is not None:
            x = (x @ w.T).astype(jnp.float32)

        if self.x_transform == "sign":
            sgn = jnp.sign(x)
            return jnp.where(sgn == 0, jnp.array(-1.0, dtype=sgn.dtype), sgn)
        elif self.x_transform == "tanh":
            return jnp.tanh(x)
        else:
            return x

    def _encode_labels(self, y: jax.Array) -> jax.Array:
        """Encode labels according to label_mode."""
        one_hot: jax.Array = jax.nn.one_hot(y, self.NUM_CLASSES, dtype=jnp.float32)
        if self.label_mode == "c-rescale":
            result: jax.Array = one_hot * (self.NUM_CLASSES**0.5 / 2.0) - 0.5
            return result
        elif self.label_mode == "pm1":
            result = one_hot * 2.0 - 1.0
            return result
        else:
            return one_hot

    def _compute_bounds(self, n: int) -> list[tuple[int, int]]:
        """Compute batch boundaries."""
        n_batches = -(-n // self.batch_size)
        return [(i * self.batch_size, min((i + 1) * self.batch_size, n)) for i in range(n_batches)]
