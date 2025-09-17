import operator
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from bionet.modules.interfaces import Adapter, Layer
from bionet.utils.typing import PyTree


class DebugLayer(Layer):
    """A simple trainable layer for testing and debugging."""

    w: Array
    a: bool = eqx.field(static=True)

    def __init__(self) -> None:
        """Initialize the weight equal to 1.0."""
        self.w = jnp.ones((1,), dtype=jnp.float32)
        self.a = True

    def __call__(self, x: Array, rng: DTypeLike | None = None) -> Array:
        """Return elementwise multiplication."""
        return self.w * x

    def activation(self, x: Array) -> Array:
        """Apply the layer’s activation function."""
        return jnp.sign(x)

    def reduce(self, h: PyTree) -> Array:
        """Aggregate incoming messages into a single tensor."""
        return jnp.asarray(jax.tree.reduce_associative(operator.add, h))

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """Return the update is the itself."""
        return self


class DebugAdapter(Adapter):
    """A stateless mapping between layers."""

    w: Array
    a: bool = eqx.field(static=True)

    def __init__(self) -> None:
        """Initialize the weight equal to 1.0."""
        self.w = jnp.ones((1,), dtype=jnp.float32)
        self.a = False

    def __call__(self, x: Array, rng: DTypeLike | None = None) -> Array:
        """Return elementwise multiplication."""
        return self.w * x

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """Return the update is the itself."""
        return self
