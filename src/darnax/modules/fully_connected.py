from __future__ import annotations

from typing import TYPE_CHECKING, Self

import equinox as eqx
import jax
import jax.numpy as jnp

from darnax.modules.interfaces import Adapter
from darnax.utils.perceptron_rule import perceptron_rule_backward

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import ArrayLike, DTypeLike

    KeyArray = Array


class FullyConnected(Adapter):
    """Fully-connected trainable adapter (x @ W, per-output scaling)."""

    W: Array
    strength: Array
    threshold: Array

    def __init__(
        self,
        in_features: int,
        out_features: int,
        strength: float | ArrayLike,
        threshold: float | ArrayLike,
        key: Array,
        dtype: DTypeLike = jnp.float32,
    ):
        """Initialize weights and per-output strength/threshold."""
        self.W = jax.random.normal(key, (in_features, out_features), dtype=dtype) / jnp.sqrt(
            in_features
        )
        self.strength = self._set_shape(strength, out_features, dtype)
        self.threshold = self._set_shape(threshold, out_features, dtype)

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Compute y = (x @ W) * strength (broadcast on last dim)."""
        return (x @ self.W) * self.strength

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """Return a module-shaped update: ΔW set; strength/threshold zero."""
        dW = perceptron_rule_backward(x, y, y_hat, self.threshold)
        zero_update = jax.tree.map(jnp.zeros_like, self)
        new_self: Self = eqx.tree_at(lambda m: m.W, zero_update, dW)
        return new_self

    @staticmethod
    def _set_shape(x: ArrayLike, dim: int, dtype: DTypeLike) -> Array:
        """Return a (dim,) vector from scalar or same-length vector."""
        x = jnp.array(x, dtype=dtype)
        if x.ndim == 0:
            return jnp.broadcast_to(x, (dim,))
        if x.ndim == 1:
            if x.shape[0] != dim:
                raise ValueError(f"length {x.shape[0]} != expected {dim}")
            return x
        raise ValueError("expected scalar or 1D vector")


class FrozenFullyConnected(FullyConnected):
    """Implements a fully connected layer with no updates.

    QOL implementation for Wback.
    """

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """Return zero update for all parameters."""
        zero_update: Self = jax.tree.map(jnp.zeros_like, self)
        return zero_update
