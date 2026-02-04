"""Tinyimagenet (precomputed features) dataset for darnax.

Loads standardized 2048-D features from Hugging Face:
    repo = "willinki/tinyimagenet-features-mlpinput"

Split semantics (HF naming quirk):
    - "train"       → training (optionally split into train/valid via `validation_fraction`)
    - "validation"  → actually the TEST set for this repo

Columns:
    - "x": features, float32 tensor of shape [N, 512]
    - "y": labels,   int32 tensor of shape [N]

Options:
    - Optional uniform-per-class subsampling on the *training* set.
    - Optional **linear projection** W ∈ ℝ^{D_out×D_in} with entries ~ N(0,1)/√D_in
      (variance-preserving when input var≈1).
    - Optional **x_transform**:
        * "identity": no change (default)
        * "sign": binarize to ±1 (0 → −1)

Post-build shapes:
    x_train: [N_tr, D_out], y_train: [N_tr, C]
    x_valid: [N_va, D_out], y_valid: [N_va, C]   (if validation_fraction > 0)
    x_test:  [N_te, D_out], y_test:  [N_te, C]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from datasets import config as hf_config  # type: ignore[import-untyped]
from datasets import load_dataset

from darnax.datasets.classification.interface import ClassificationDataset, RescalingMode

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class TinyImagenetFeatures(ClassificationDataset):
    """TinyImageNet features dataset (512-D, standardized)."""

    NUM_CLASSES = 200
    DEFAULT_RESCALING = "null"
    FEAT_DIM = 512
    SHAPE_DIM = 2
    HF_REPO = "willinki/tinyimagenet-features"
    CACHE_SUBDIR = "darnax/tinyimagenet-features"

    def __init__(
        self,
        batch_size: int = 64,
        linear_projection: int | None = None,
        num_images_per_class: int | None = None,
        label_mode: Literal["pm1", "ooe", "c-rescale"] = "c-rescale",
        x_transform: Literal["sign", "identity"] = "identity",
        validation_fraction: float = 0.0,
        rescaling: RescalingMode = "default",
    ) -> None:
        """Initialize Cifar10Features data.

        Takes care of downloading the tensors in jax format, sampling with
        equal frequency, sample a validation set and transforming x or y when
        requested.

        Parameters
        ----------
        batch_size : int
            Batch size for the iterators
        linear_projection : int, optional
            If integer, data is linearly projected to the size. If None, the
            step is skipped.
        num_images_per_class : int, optional
            If integer, we simple a fixed amount of images per class. If a
            class does not contain enough images, we sample them all.
        label_mode : Literal["pm1", "ooe", "c-rescale"]
            if pm1 the positive class is assigned +1, the others -1. if ooe,
            regular one-hot-encoding. if c-rescale, the positive class is
            rescaled to C/2, while the negative are rescaled to -0.5.
        validation_fraction : float
            If not zero, we sample a random holdout set from training.
        x_transform : Literal["sign", "identity"]
            if sign, we binarize features after linear transform. if identity
            this step is skipped.
        rescaling : RescalingMode
            Rescaling mode: "default" (no change, features pre-standardized),
            "null" (no rescaling), "divide255" (divide by 255),
            "standardize" (mean=0, std=1).

        Raises
        ------
        ValueError
            if batch_size <= 1, linear_projection is not int >= 1. if
            num_images_per_class is negative or not None. if
            validation_fraction is outside [0.0, 1.0).

        """
        if batch_size <= 1:
            raise ValueError(f"Invalid batch_size={batch_size!r}; must be > 1.")
        if linear_projection is not None and (
            not isinstance(linear_projection, int) or linear_projection <= 0
        ):
            raise ValueError("`linear_projection` must be a positive int or None.")
        if num_images_per_class is not None and num_images_per_class <= 0:
            raise ValueError("`num_images_per_class` must be positive or None.")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("`validation_fraction` must be in [0.0, 1.0).")

        self.batch_size = int(batch_size)
        self.linear_projection = linear_projection
        self.num_images_per_class = num_images_per_class
        self.label_mode = label_mode
        self.validation_fraction = validation_fraction
        self.x_transform = x_transform
        self.rescaling = rescaling

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

    # ----------------------------- Public API -----------------------------

    def build(self, key: jax.Array) -> jax.Array:
        """Load, optionally subsample, optionally project, transform, encode labels, split."""
        key_sample, key_proj, key_split, key_shuf, rng = jax.random.split(key, 5)

        # HF "validation" split is the TEST set for this repo.
        x_tr_all, y_tr_all = self._load_split("train")
        x_te_all, y_te_all = self._load_split("validation")

        # Optional uniform-per-class subsample on training only.
        if self.num_images_per_class is None:
            x_tr, y_tr = x_tr_all, y_tr_all
        else:
            x_tr, y_tr = self._subsample_per_class(
                key_sample, x_tr_all, y_tr_all, self.num_images_per_class
            )

        # Optional holdout validation from the (possibly subsampled) training set.
        if self.validation_fraction > 0.0:
            n_total = x_tr.shape[0]
            n_valid = int(n_total * self.validation_fraction)
            perm = jax.random.permutation(key_split, n_total)
            x_tr, y_tr = x_tr[perm], y_tr[perm]
            x_tr, x_va = x_tr[:-n_valid], x_tr[-n_valid:]
            y_tr, y_va = y_tr[:-n_valid], y_tr[-n_valid:]
        else:
            x_va, y_va = None, None

        # Apply rescaling.
        x_tr = self._apply_rescaling(x_tr)
        x_te_all = self._apply_rescaling(x_te_all)
        if x_va is not None:
            x_va = self._apply_rescaling(x_va)

        # Optional linear projection with 1/sqrt(in_dim) scaling.
        w = (
            self._generate_random_projection(key_proj, int(self.linear_projection), self.FEAT_DIM)
            if self.linear_projection is not None
            else None
        )
        x_tr = self._apply_projection(w, x_tr)
        x_te = self._apply_projection(w, x_te_all)
        if x_va is not None:
            x_va = self._apply_projection(w, x_va)

        # Optional x transform (e.g., sign binarization).
        x_tr = self._apply_x_transform(x_tr)
        x_te = self._apply_x_transform(x_te)
        if x_va is not None:
            x_va = self._apply_x_transform(x_va)

        # Encode labels.
        y_tr_enc = self._encode_labels(y_tr)
        y_te_enc = self._encode_labels(y_te_all)
        y_va_enc = self._encode_labels(y_va) if y_va is not None else None

        # Shuffle training set for iteration.
        perm = jax.random.permutation(key_shuf, x_tr.shape[0])
        self.x_train, self.y_train = x_tr[perm], y_tr_enc[perm]

        # Test and (optional) validation.
        self.x_test, self.y_test = x_te, y_te_enc
        if x_va is not None and y_va_enc is not None:
            self.x_valid, self.y_valid = x_va, y_va_enc

        # Specs and sanity.
        self.input_dim = int(self.x_train.shape[1])
        if self.linear_projection is None and self.input_dim != self.FEAT_DIM:
            logger.warning("Detected feature dim %d (expected %d).", self.input_dim, self.FEAT_DIM)

        # Batch bounds
        self._train_bounds = self._compute_bounds(self.x_train.shape[0])
        self._test_bounds = self._compute_bounds(self.x_test.shape[0])
        if self.x_valid is not None:
            self._valid_bounds = self._compute_bounds(self.x_valid.shape[0])

        rng_out: jax.Array = rng
        return rng_out

    def __iter__(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Iterate over training batches."""
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Dataset not built. Call `build()` first.")
        for lo, hi in self._train_bounds:
            yield self.x_train[lo:hi], self.y_train[lo:hi]

    def iter_test(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Iterate over test batches (HF 'validation' split)."""
        if self.x_test is None or self.y_test is None:
            raise RuntimeError("Dataset not built. Call `build()` first.")
        for lo, hi in self._test_bounds:
            yield self.x_test[lo:hi], self.y_test[lo:hi]

    def iter_valid(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Iterate over validation batches (holdout from training)."""
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
        """Return dataset specification for model wiring."""
        if self.x_train is None or self.y_train is None or self.input_dim is None:
            raise RuntimeError("Dataset not built. Call `build()` first.")
        return {
            "x_shape": (self.input_dim,),
            "x_dtype": self.x_train.dtype,
            "y_shape": (self.num_classes,),
            "y_dtype": self.y_train.dtype,
            "num_classes": self.num_classes,
            "label_encoding": self.label_mode,
            "projected_dim": self.input_dim if self.linear_projection else None,
        }

    # --------------------------- Internal Helpers ---------------------------

    def _load_split(self, split: str) -> tuple[jax.Array, jax.Array]:
        """Load a split and return (x, y) as JAX tensors.

        Expects exact columns:
            - 'x': float32 features of shape [N, FEAT_DIM]
            - 'y': int32 labels of shape [N]
        """
        data_naxis = 2
        x_np, y_np = type(self)._load_split_numpy(split)
        x = jnp.asarray(x_np, dtype=jnp.float32)
        y = jnp.asarray(y_np, dtype=jnp.int32)
        if x.ndim != data_naxis or x.shape[1] != self.FEAT_DIM:
            raise ValueError(f"Expected x shape [N,{self.FEAT_DIM}], got {tuple(x.shape)}.")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("Label vector must be [N] and match features.")
        return x, y

    @staticmethod
    def _generate_random_projection(key: jax.Array, out_dim: int, in_dim: int) -> jax.Array:
        """Generate Gaussian projection matrix with variance-preserving scaling.

        Returns W ∈ ℝ^{out_dim × in_dim} with entries ~ N(0, 1/ in_dim),
        implemented as N(0,1)/√in_dim to keep output variance ≈ 1
        when input features have variance ≈ 1.
        """
        return jax.random.normal(key, (out_dim, in_dim), dtype=jnp.float32) / jnp.sqrt(in_dim)

    @staticmethod
    def _apply_projection(w: jax.Array | None, x: jax.Array) -> jax.Array:
        """Apply linear projection if provided; otherwise return x."""
        if w is None:
            return x
        # x: [N, in_dim], w: [out_dim, in_dim] → [N, out_dim]
        return (x @ w.T).astype(jnp.float32)

    def _apply_x_transform(self, x: jax.Array) -> jax.Array:
        """Apply the configured x transform."""
        if self.x_transform == "sign":
            sgn = jnp.sign(x)
            return jnp.where(sgn == 0, jnp.array(-1.0, dtype=sgn.dtype), sgn)
        # identity
        return x

    @classmethod
    def _subsample_per_class(
        cls, key: jax.Array, x: jax.Array, y: jax.Array, k: int
    ) -> tuple[jax.Array, jax.Array]:
        """Uniform-by-class sampling of up to k examples per class."""
        xs, ys = [], []
        for c in range(cls.NUM_CLASSES):
            key, sub = jax.random.split(key)
            idx = jnp.where(y == c)[0]
            n = min(k, int(idx.shape[0]))
            if n == 0:
                continue
            perm = jax.random.permutation(sub, idx.shape[0])
            xs.append(x[idx[perm[:n]]])
            ys.append(y[idx[perm[:n]]])
        if not xs:
            raise ValueError("Requested per-class subsample produced no data.")
        return jnp.concatenate(xs), jnp.concatenate(ys)

    def _encode_labels(self, y: jax.Array) -> jax.Array:
        """Encode labels according to label_mode."""
        one_hot: jax.Array = jax.nn.one_hot(y, self.NUM_CLASSES, dtype=jnp.float32)
        if self.label_mode == "c-rescale":
            rescaled_y: jax.Array = one_hot * (self.NUM_CLASSES**0.5 / 2.0) - 0.5
            return rescaled_y
        elif self.label_mode == "pm1":
            rescaled_y_pm1: jax.Array = one_hot * 2.0 - 1.0
            return rescaled_y_pm1
        else:
            return one_hot

    def _compute_bounds(self, n: int) -> list[tuple[int, int]]:
        """Compute [lo, hi) batch boundaries."""
        n_batches = -(-n // self.batch_size)
        return [(i * self.batch_size, min((i + 1) * self.batch_size, n)) for i in range(n_batches)]

    @classmethod
    def _load_split_numpy(cls, split: str) -> tuple[np.ndarray, np.ndarray]:
        cache_file = cls._preprocessed_cache_path(split)
        if cache_file is not None and cache_file.exists():
            try:
                with np.load(cache_file, allow_pickle=False) as data:
                    x_np = data["x"]
                    y_np = data["y"]
            except Exception:  # pragma: no cover - fall back when cache is corrupt
                logger.warning(
                    "Failed to read cached CIFAR-10 features split %s; regenerating.", split
                )
                x_np, y_np = cls._generate_and_cache_split(split, cache_file)
        else:
            x_np, y_np = cls._generate_and_cache_split(split, cache_file)
        return x_np, y_np

    @classmethod
    def _generate_and_cache_split(
        cls, split: str, cache_file: Path | None
    ) -> tuple[np.ndarray, np.ndarray]:
        ds = load_dataset(cls.HF_REPO, split=split)
        if "x" not in ds.column_names or "y" not in ds.column_names:
            raise KeyError(
                f"Split {split!r} must contain 'x' and 'y' columns. Found: {ds.column_names}."
            )
        ds.set_format(type="numpy", columns=["x", "y"])
        batch = ds[:]
        x_np = batch["x"].astype(np.float32)
        y_np = batch["y"].astype(np.int32)

        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = cache_file.with_suffix(".tmp.npz")
                np.savez(tmp_path, x=x_np, y=y_np)
                os.replace(tmp_path, cache_file)
            except Exception:  # pragma: no cover - cache failures should not block loading
                logger.warning("Unable to write CIFAR-10 features cache for split %s", split)

        return x_np, y_np

    @classmethod
    def _preprocessed_cache_path(cls, split: str) -> Path | None:
        base = os.environ.get("DARNAX_DATASET_CACHE")
        if base is None:
            base = os.environ.get("HF_DATASETS_CACHE")
        if base is None:
            base = getattr(hf_config, "HF_DATASETS_CACHE", None)
        if base is None:
            return None
        cache_root = Path(base) / cls.CACHE_SUBDIR
        return cache_root / f"{split}.npz"
