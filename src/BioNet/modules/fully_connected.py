import jax
import jax.numpy as jnp
from jax.typing import Array, ArrayLike, PyTree
from src.modules.interfaces import Adapter


class FullyConnected(Adapter):
    """Ferromagnetic adapter"""

    W: Array
    strength: Array
    threshold: Array

    def __init__(
        self,
        in_features: int,
        out_features: int,
        strength: float,
        threshold: float,
        key: Array,
        dtype=jnp.float32,
    ):
        self.W = jax.random.normal(
            key, (in_features, out_features), dtype=dtype
        ) / jnp.sqrt(in_features)
        self.strength = self._set_shape(strength, out_features, dtype)
        self.threshold = self._set_shape(threshold, out_features, dtype)

    @jax.jit
    def __call__(self, x: Array) -> Array:
        return x @ self.W * self.strength

    @jax.jit
    def backward(self, s: Array, h: Array) -> "FullyConnected":

    @staticmethod
    def _set_shape(x: ArrayLike, dim: int, dtype):
        x = jnp.array(x, dtype=dtype)
        assert len(x.shape) == 0 or len(x.shape) == 1
        return jnp.broadcast_to(x, shape=(dim,))


# todo: define one with with init = diagonal set to strength and the rest
# to 1/sqrt(n)
