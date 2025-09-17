from typing import Self

import jax.numpy as jnp
from jax import Array
from jax.tree_util import tree_map
from jax.typing import ArrayLike, DTypeLike

from bionet.modules.interfaces import Adapter

KeyArray = Array


class Ferromagnetic(Adapter):
    """Elementwise scaling adapter with fixed coupling strength."""

    strength: Array

    def __init__(self, features: int, strength: ArrayLike, dtype: DTypeLike = jnp.float32):
        """Initialize the adapter with scalar or per-feature strength."""
        self.strength = self._set_shape(strength, features, dtype)

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Scale inputs elementwise by the coupling strength."""
        return x * self.strength

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """Return a zero update to indicate non-trainability."""
        new_self: Self = tree_map(jnp.zeros_like, self)
        return new_self

    @staticmethod
    def _set_shape(x: ArrayLike, features: int, dtype: DTypeLike) -> Array:
        """Return a (features,) strength vector from scalar or vector input."""
        x = jnp.array(x, dtype)
        if x.ndim == 0:
            return jnp.broadcast_to(x, (features,))
        if x.ndim == 1:
            if x.shape[0] != features:
                raise ValueError(f"strength length {x.shape[0]} != features {features}")
            return x
        raise ValueError("strength must be a scalar or 1D vector")
