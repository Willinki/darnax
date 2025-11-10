"""Fashion-MNIST dataset implementation for darnax.

This module defines :class:`FashionMnist`, a light wrapper around the Hugging Face
``fashion_mnist`` dataset that provides standardized preprocessing, label encoding,
optional random linear projection, and iterators for train/validation/test splits.

Notes
-----
- Images are loaded as float32 in ``[0, 1]`` and flattened to ``(N, 784)`` before
  optional projection.
- Labels can be encoded as ±1, one-of-K, or a centered/rescaled variant.
- A deterministic shuffle is applied to the training split during ``build()``.
- Validation data is an optional holdout taken from the *training* set.
- Iterators yield mini-batches of tensors shaped ``(B, D)`` (inputs) and
  ``(B, C)`` (targets).

Examples
--------
Basic usage

>>> import jax, jax.numpy as jnp
>>> from darnax.datasets.classification.fashion_mnist import FashionMnist
>>> ds = FashionMnist(batch_size=128, linear_projection=100, x_transform="sign")
>>> ds.build(jax.random.PRNGKey(0))
>>> x, y = next(iter(ds))
>>> x.shape, y.shape
((128, ds.input_dim), (128, ds.NUM_CLASSES))

Spec dictionary

>>> spec = ds.spec()
>>> sorted(spec.keys())
['label_encoding', 'num_classes', 'projected_dim', 'x_dtype', 'x_shape', 'y_dtype', 'y_shape']

"""

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


class FashionMnist(ClassificationDataset):
    """Fashion-MNIST dataset with configurable preprocessing.

    This dataset loader fetches splits from Hugging Face (``fashion_mnist``), applies
    optional random linear projection with variance-preserving scaling
    ``1/sqrt(in_dim)``, and supports three input transforms (``sign``, ``tanh``,
    ``identity``). Labels can be encoded in three modes and a validation split can be
    held out from the training set.

    Parameters
    ----------
    batch_size : int, default=64
        Mini-batch size produced by the iterators. Must be > 1.
    linear_projection : int or None, default=100
        Output dimensionality ``D`` of a shared random Gaussian projection applied to
        flattened images (784 → ``D``). If ``None``, no projection is applied.
    num_images_per_class : int or None, default=None
        If set, subsamples at most ``k`` training images per class (uniformly at
        random, class-balanced). Uses all training data when ``None``.
    label_mode : {"pm1", "ooe", "c-rescale"}, default="c-rescale"
        Label encoding scheme:
        - ``"pm1"``: ±1 one-vs-all targets.
        - ``"ooe"``: one-of-K (standard one-hot, 1 for class, 0 otherwise).
        - ``"c-rescale"``: centered/rescaled one-hot,
        ``one_hot * sqrt(C)/2 - 1/2``, where ``C`` is the number of classes.
    x_transform : {"sign", "tanh", "identity"}, default="sign"
        Input nonlinearity applied *after* optional projection:
        - ``"sign"`` maps zeros to ``-1`` for strict binary {−1, +1}.
        - ``"tanh"`` applies ``tanh`` element-wise.
        - ``"identity"`` leaves features unchanged.
    validation_fraction : float, default=0.0
        Fraction of the *training* set to reserve as validation holdout
        (``0.0 <= fraction < 1.0``). When ``0.0``, no validation split is created.

    Attributes
    ----------
    NUM_CLASSES : int
        Number of classes (10).
    FLAT_DIM : int
        Flattened pixel dimension (28 * 28 = 784).
    batch_size : int
        Effective batch size used by iterators.
    linear_projection : int | None
        Target projected dimension or ``None`` if disabled.
    num_images_per_class : int | None
        Per-class cap for training subsampling, if set.
    label_mode : Literal["pm1","ooe","c-rescale"]
        Encoding mode applied by :meth:`_encode_labels`.
    x_transform : Literal["sign","tanh","identity"]
        Input transform applied by :meth:`_preprocess`.
    validation_fraction : float
        Ratio of training data held out as validation.
    input_dim : int | None
        Feature dimensionality after preprocessing; set by :meth:`build`.
    num_classes : int
        Number of classes (equals ``NUM_CLASSES``).
    x_train, y_train, x_valid, y_valid, x_test, y_test : jax.Array | None
        In-memory arrays for each split, populated by :meth:`build`.

    """

    NUM_CLASSES = 10
    FLAT_DIM = 28 * 28

    def __init__(
        self,
        batch_size: int = 64,
        linear_projection: int | None = 100,
        num_images_per_class: int | None = None,
        label_mode: Literal["pm1", "ooe", "c-rescale"] = "c-rescale",
        x_transform: Literal["sign", "tanh", "identity"] = "sign",
        validation_fraction: float = 0.0,
    ) -> None:
        """Initialize the dataset object."""
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

    # ------------------------------- Build ------------------------------- #

    def build(self, key: jax.Array) -> None:
        """Load data, preprocess, encode labels, and prepare iterators.

        This method:
        1) Loads raw train/test splits from Hugging Face.
        2) Optionally subsamples training per class.
        3) Optionally creates a validation holdout from the training set.
        4) Applies a shared random linear projection (if enabled).
        5) Applies the chosen input transform.
        6) Encodes labels per ``label_mode``.
        7) Deterministically shuffles training data.
        8) Builds batch bounds for all available splits.

        Parameters
        ----------
        key : jax.Array
            PRNG key. Four splits are derived internally for sampling, projection
            weights, train/valid split permutation, and train shuffle.

        """
        key_sample, key_proj, key_split, key_shuf = jax.random.split(key, 4)

        x_tr_all, y_tr_all = self._load_split("train")
        x_te_all, y_te_all = self._load_split("test")

        if self.num_images_per_class is None:
            x_tr, y_tr = x_tr_all, y_tr_all
        else:
            x_tr, y_tr = self._subsample_per_class(
                key_sample, x_tr_all, y_tr_all, self.num_images_per_class
            )

        # Optional validation holdout from TRAIN
        if self.validation_fraction > 0.0:
            n_total = x_tr.shape[0]
            n_valid = int(n_total * self.validation_fraction)
            perm = jax.random.permutation(key_split, n_total)
            x_tr, y_tr = x_tr[perm], y_tr[perm]
            x_tr, x_va = x_tr[:-n_valid], x_tr[-n_valid:]
            y_tr, y_va = y_tr[:-n_valid], y_tr[-n_valid:]
        else:
            x_va, y_va = None, None

        # Shared projection (if any), flatten always
        w = (
            self._generate_random_projection(key_proj, int(self.linear_projection), self.FLAT_DIM)
            if self.linear_projection is not None
            else None
        )

        x_tr = self._preprocess(w, x_tr)
        x_te = self._preprocess(w, x_te_all)
        if x_va is not None:
            x_va = self._preprocess(w, x_va)

        # Labels
        y_tr = self._encode_labels(y_tr)
        y_te = self._encode_labels(y_te_all)
        y_va = self._encode_labels(y_va) if y_va is not None else None

        # Deterministic train shuffle
        perm = jax.random.permutation(key_shuf, x_tr.shape[0])
        self.x_train, self.y_train = x_tr[perm], y_tr[perm]
        self.x_test, self.y_test = x_te, y_te
        if x_va is not None and y_va is not None:
            self.x_valid, self.y_valid = x_va, y_va

        self.input_dim = int(self.x_train.shape[1])

        # Batch ranges
        self._train_bounds = self._compute_bounds(self.x_train.shape[0])
        self._test_bounds = self._compute_bounds(self.x_test.shape[0])
        if self.x_valid is not None:
            self._valid_bounds = self._compute_bounds(self.x_valid.shape[0])

    # ------------------------------- Iterators ------------------------------- #

    def __iter__(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Iterate over training mini-batches.

        Yields
        ------
        tuple[jax.Array, jax.Array]
            A pair ``(x_batch, y_batch)`` with shapes ``(B, D)`` and ``(B, C)``.

        """
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Dataset not built. Call `build()` first.")
        for lo, hi in self._train_bounds:
            yield self.x_train[lo:hi], self.y_train[lo:hi]

    def iter_test(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Iterate over test mini-batches.

        Yields
        ------
        tuple[jax.Array, jax.Array]
            A pair ``(x_batch, y_batch)`` from the test split.

        """
        if self.x_test is None or self.y_test is None:
            raise RuntimeError("Dataset not built. Call `build()` first.")
        for lo, hi in self._test_bounds:
            yield self.x_test[lo:hi], self.y_test[lo:hi]

    def iter_valid(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Iterate over validation mini-batches.

        Raises
        ------
        NotImplementedError
            If no validation split was created (``validation_fraction == 0``).

        Yields
        ------
        tuple[jax.Array, jax.Array]
            A pair ``(x_batch, y_batch)`` from the validation split.

        """
        if self.x_valid is None or self.y_valid is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} has no validation split. "
                "Set validation_fraction > 0 to create one."
            )
        for lo, hi in self._valid_bounds:
            yield self.x_valid[lo:hi], self.y_valid[lo:hi]

    def __len__(self) -> int:
        """Return number of training mini-batches.

        Returns
        -------
        int
            The number of batches produced by the training iterator.

        Raises
        ------
        RuntimeError
            If the dataset was not yet built.

        """
        if not self._train_bounds:
            raise RuntimeError("Dataset not built. Call `build()` first.")
        return len(self._train_bounds)

    def spec(self) -> dict[str, Any]:
        """Return a lightweight spec describing shapes, dtypes, and encoding.

        Returns
        -------
        dict[str, Any]
            Keys:
            - ``"x_shape"``: tuple[int], single-example input shape (``(D,)``).
            - ``"x_dtype"``: dtype of inputs.
            - ``"y_shape"``: tuple[int], single-example target shape (``(C,)``).
            - ``"y_dtype"``: dtype of targets.
            - ``"num_classes"``: number of classes (10).
            - ``"label_encoding"``: the selected label mode.
            - ``"projected_dim"``: ``D`` if projection enabled, else ``None``.

        Raises
        ------
        RuntimeError
            If called before :meth:`build` has initialized arrays.

        """
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

    # ------------------------------- Internals ------------------------------- #

    @staticmethod
    def _load_split(split: str) -> tuple[jax.Array, jax.Array]:
        """Load a split from Hugging Face and convert to tensors.

        Parameters
        ----------
        split : str
            Either ``"train"`` or ``"test"``.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            ``(x, y)`` where ``x`` is ``float32`` in ``[0, 1]`` with shape
            ``(N, 28, 28)`` and ``y`` is ``int32`` with shape ``(N,)``.

        """
        ds = load_dataset("fashion_mnist", split=split, trust_remote_code=True)
        x_raw = jnp.asarray([jnp.array(im) for im in ds["image"]], dtype=jnp.float32)
        x: jax.Array = (x_raw / 255.0).astype(jnp.float32)
        y: jax.Array = jnp.asarray(ds["label"], dtype=jnp.int32)
        return x, y

    @classmethod
    def _subsample_per_class(
        cls, key: jax.Array, x: jax.Array, y: jax.Array, k: int
    ) -> tuple[jax.Array, jax.Array]:
        """Uniformly subsample up to ``k`` items per class from ``(x, y)``.

        Parameters
        ----------
        key : jax.Array
            PRNG key used for class-wise permutations.
        x : jax.Array
            Input images, shape ``(N, H, W)`` or ``(N, D)``.
        y : jax.Array
            Integer labels, shape ``(N,)``.
        k : int
            Maximum examples to keep per class.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            Subsampled ``(x_sub, y_sub)`` concatenated over classes.

        """
        xs, ys = [], []
        for c in range(cls.NUM_CLASSES):
            key, sub = jax.random.split(key)
            idx = jnp.where(y == c)[0]
            n = min(k, int(idx.shape[0]))
            perm = jax.random.permutation(sub, idx.shape[0])
            xs.append(x[idx[perm[:n]]])
            ys.append(y[idx[perm[:n]]])
        return jnp.concatenate(xs), jnp.concatenate(ys)

    @staticmethod
    def _generate_random_projection(key: jax.Array, out_dim: int, in_dim: int) -> jax.Array:
        """Create a Gaussian projection matrix with ``1/sqrt(in_dim)`` scaling.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        out_dim : int
            Number of output features (rows).
        in_dim : int
            Number of input features (columns).

        Returns
        -------
        jax.Array
            Weight matrix ``W ∈ R^{out_dim×in_dim}`` with entries
            ``N(0, 1) / sqrt(in_dim)`` in ``float32``.

        """
        return jax.random.normal(key, (out_dim, in_dim), dtype=jnp.float32) / jnp.sqrt(in_dim)

    def _preprocess(self, w: jax.Array | None, x: jax.Array) -> jax.Array:
        """Flatten, optionally project, and apply the configured transform.

        Parameters
        ----------
        w : jax.Array or None
            Projection matrix ``(D, 784)``. If ``None``, no projection is applied.
        x : jax.Array
            Inputs shaped ``(N, 28, 28)`` or already flat; cast to ``(N, D_in)``.

        Returns
        -------
        jax.Array
            Preprocessed features shaped ``(N, D)``.

        """
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
        """Encode integer labels according to ``label_mode``.

        Parameters
        ----------
        y : jax.Array
            Integer class indices, shape ``(N,)``.

        Returns
        -------
        jax.Array
            Encoded targets, shape ``(N, C)`` with ``C = NUM_CLASSES``.

        """
        one_hot: jax.Array = jax.nn.one_hot(y, self.NUM_CLASSES, dtype=jnp.float32)
        if self.label_mode == "c-rescale":
            rescaled_y: jax.Array = one_hot * (self.NUM_CLASSES**0.5 / 2.0) - 0.5
            return rescaled_y
        elif self.label_mode == "pm1":
            rescaled_pm1_y: jax.Array = one_hot * 2.0 - 1.0
            return rescaled_pm1_y
        else:
            return one_hot

    def _compute_bounds(self, n: int) -> list[tuple[int, int]]:
        """Compute half-open index ranges ``[lo, hi)`` for batching ``n`` items.

        Parameters
        ----------
        n : int
            Number of examples.

        Returns
        -------
        list[tuple[int, int]]
            Consecutive ranges covering ``0..n`` with step ``batch_size``.

        """
        n_batches = -(-n // self.batch_size)  # ceil div
        return [(i * self.batch_size, min((i + 1) * self.batch_size, n)) for i in range(n_batches)]
