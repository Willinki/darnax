from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Self

import jax

from darnax.modules.interfaces import Layer

if TYPE_CHECKING:
    from jax import Array

    from darnax.utils.typing import PyTree

    KeyArray = Array


class OutputLayer(Layer):
    """Simple identity output layer that sums predictions.

    This layer leaves inputs unchanged on forward/activation, and provides a
    `reduce` method that elementwise-sums all array leaves in a PyTree of
    predictions. The backward pass is a no-op because this layer has no
    trainable parameters.
    """

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Return the input unchanged (identity forward)."""
        return jax.numpy.zeros_like(x)

    def reduce(self, h: PyTree) -> Array:
        """Elementwise-sum all array leaves in a PyTree of predictions.

        Parameters
        ----------
        h
            PyTree whose leaves are arrays with identical shapes/dtypes.

        Returns
        -------
        Array
            The elementwise sum across all leaves.

        Raises
        ------
        ValueError
            If `h` has no leaves.

        """
        return jax.numpy.asarray(jax.tree_util.tree_reduce(operator.add, h))

    def activation(self, x: Array) -> Array:
        """Identity activation."""
        return x

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """No-op backward because the output layer has no parameters."""
        return self
