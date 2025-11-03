from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator, Tuple
import jax
import jax.numpy as jnp

class AbstractDataset(ABC):
    """
    NOTES
    - Preprocessing, batching and optional validation splitting happen inside `build()`.
    - A validation split is optional. If not used, iter_valid() can yield nothing.
    """
    @abstractmethod
    def build(self, key: jax.Array) -> jax.Array:
        """Load, preprocess, and prepare dataset splits.

        Responsibilities
        ----------------
        - Load raw data and apply preprocessing.
        - Optionally create a validation split .
        - Prepare batching.
        - Return a new PRNG key.

        Parameters
        ----------
        key : jax.Array
            PRNG key for deterministic sampling and preprocessing.

        Returns
        -------
        jax.Array
            New PRNG key after use.
        """
        pass
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[jax.Array, jax.Array]]:
        """Iterate over training batches."""
        pass

    @abstractmethod
    def iter_valid(self) -> Iterator[Tuple[jax.Array, jax.Array]]:
        """Iterate over validation batches.
        Note: Kept it as an @abstractmethod for consistency so every subclass explicitly defines its behavior, we can make it return an empty iterator if no validation split.
        """
        pass

    @abstractmethod
    def iter_test(self) -> Iterator[Tuple[jax.Array, jax.Array]]:
        """Iterate over test batches."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of training batches"""
        pass

    @abstractmethod
    def spec(self) -> dict:

        """Describe what each sample looks like.

        Returns
        -------
        dict
            Example keys:
              - 'x_shape': input sample shape
              - 'y_shape': target shape
              - 'num_classes': number of classes
              - 'x_dtype', 'y_dtype': JAX dtypes
              - 'label_encoding': e.g. 'ooe', 'pm1', 'c-rescale'
        """  
        pass