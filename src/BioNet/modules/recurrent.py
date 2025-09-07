import functools
import operator
import jax
import jax.numpy as jnp
from jax.typing import Array, PyTree, ArrayLike
import equinox as eqx
from src.modules.interfaces import Layer


class RecurrentDiscrete(Layer):
    """Simple +-1 recurrent layer"""

    J: Array
    J_D: Array
    threshold: Array

    def __init__(
        self,
        features: int,
        J_D: ArrayLike,
        threshold: ArrayLike,
        key: Array,
        dtype=jnp.float32,
    ):
        J_D = self._set_shape(J_D, features, dtype)
        threshold = self._set_shape(threshold, features)
        J = jax.random.normal(key, shape=(features, features), dtype=dtype) / jnp.sqrt(
            features
        )
        diag_indices = jnp.diag_indices(features)
        J = J.at[diag_indices].set(J_D)
        self.J = J
        self.J_D = J_D
        self.threshold = threshold

    @jax.jit
    def activation(self, x: Array) -> Array:
        return jnp.sign(x)

    @jax.jit
    def __call__(self, x: Array) -> Array:
        return x @ self.J

    @jax.jit
    def reduce(self, h: PyTree) -> Array:
        leaves = jax.tree_util.tree_leaves(h)
        return functools.reduce(operator.add, leaves)

    @jax.jit
    def backward(self, s: Array, h: Array) -> "RecurrentDiscrete":
        delta = jnp.outer(s, s)
        mask = jnp.where(jnp.broadcast_to())
        delta =
        return eqx.tree_at(
            lambda m: (m.J, m.J_D), self, (delta, jnp.zeros_like(self.J_D))
        )

    @staticmethod
    def _set_shape(x: ArrayLike, dim: int, dtype):
        x = jnp.array(x, dtype)
        assert len(x.shape) == 0 or len(x.shape) == 1
        return jnp.broadcast_to(x, shape=(dim,))
