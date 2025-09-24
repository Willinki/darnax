from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import tree_reduce

from darnax.modules.interfaces import Layer
from darnax.utils.perceptron_rule import perceptron_rule_backward

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import ArrayLike, DTypeLike

    from darnax.utils.typing import PyTree

    KeyArray = jax.Array


class RecurrentDiscrete(Layer):
    """Binary (±1) recurrent layer with dense couplings."""

    J: Array
    J_D: Array
    threshold: Array
    _mask: Array

    def __init__(
        self,
        features: int,
        j_d: ArrayLike,
        threshold: ArrayLike,
        key: KeyArray,
        dtype: DTypeLike = jnp.float32,
    ):
        """Initialize J with Gaussian entries, set diag to j_d, and store thresholds."""
        j_d_vec = self._set_shape(j_d, features, dtype)
        thresh_vec = self._set_shape(threshold, features, dtype)

        J = jax.random.normal(key, shape=(features, features), dtype=dtype) / jnp.sqrt(features)
        diag = jnp.diag_indices(features)
        J = J.at[diag].set(j_d_vec)

        self.J = J
        self.J_D = j_d_vec
        self.threshold = thresh_vec
        self._mask = 1 - jnp.eye(features, dtype=dtype)

    def activation(self, x: Array) -> Array:
        """Return strict ±1 activation with ties mapped to +1."""
        return jnp.where(x >= 0, 1, -1).astype(x.dtype)

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Compute pre-activation h = x @ J."""
        return x @ self.J

    def reduce(self, h: PyTree) -> Array:
        """Aggregate incoming messages by summation."""
        return jnp.asarray(tree_reduce(operator.add, h))

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """Return a module-shaped update with ΔJ in J and zeros elsewhere."""
        dJ = perceptron_rule_backward(x, y, y_hat, self.threshold)
        dJ = dJ * self._mask
        zero_update = jax.tree.map(jnp.zeros_like, self)
        new_self: Self = eqx.tree_at(lambda m: m.J, zero_update, dJ)
        return new_self

    @staticmethod
    def _set_shape(x: ArrayLike, dim: int, dtype: DTypeLike) -> Array:
        """Return a (dim,) vector from scalar or same-length vector, with dtype."""
        x = jnp.array(x, dtype)
        if x.ndim == 0:
            return jnp.broadcast_to(x, (dim,))
        if x.ndim == 1:
            if x.shape[0] != dim:
                raise ValueError(f"length {x.shape[0]} != features {dim}")
            return x
        raise ValueError("expected scalar or 1D vector")
