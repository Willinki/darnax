import jax
import jax.numpy as jnp
from jax.typing import Array, ArrayLike, PyTree
from src.modules.interfaces import Adapter


class Ferromagnetic(Adapter):
    """Ferromagnetic adapter"""

    strength: Array

    def __init__(self, features: int, strength: float, dtype=jnp.float32):
        self.strength = self._set_shape(strength, features, dtype)

    @jax.jit
    def __call__(self, x: Array) -> Array:
        return x * self.strength

    @jax.jit
    def backward(self, s: Array, h: PyTree) -> "Ferromagnetic":
        return jax.tree.map(jnp.zeros_like, self)

    @staticmethod
    def _set_shape(x: ArrayLike, dim: int, dtype):
        x = jnp.array(x, dtype)
        assert len(x.shape) == 0 or len(x.shape) == 1
        return jnp.broadcast_to(x, shape=(dim,))
