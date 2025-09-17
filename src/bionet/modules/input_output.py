from __future__ import annotations

from typing import TYPE_CHECKING, Self

import jax
import jax.numpy as jnp

from bionet.modules.interfaces import Layer

if TYPE_CHECKING:
    from jax import Array

    from bionet.utils.typing import PyTree

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
        return x

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
        leaves = jax.tree_util.tree_leaves(h)
        if not leaves:
            raise ValueError("reduce() requires at least one prediction leaf.")

        # Optional shape guard; drop if you prefer to let broadcasting error naturally.
        ref_shape = leaves[0].shape
        for leaf in leaves[1:]:
            if leaf.shape != ref_shape:
                raise ValueError("All prediction leaves must have the same shape.")

        return jnp.sum(jnp.stack(leaves, axis=0), axis=0)

    def activation(self, x: Array) -> Array:
        """Identity activation."""
        return x

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """No-op backward because the output layer has no parameters."""
        return self
