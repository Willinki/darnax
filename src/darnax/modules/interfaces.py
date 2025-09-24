from abc import ABC, abstractmethod
from typing import Self

import equinox as eqx
from jax import Array

from darnax.utils.typing import PyTree

KeyArray = Array


class AbstractModule(eqx.Module, ABC):
    """Base class for layers and adapters."""

    @property
    @abstractmethod
    def has_state(self) -> bool:
        """Return whether the module carries persistent state."""
        ...

    @abstractmethod
    def __call__(self, x: Array, rng: Array | None = None) -> Array:
        """Compute the forward pass."""
        ...

    @abstractmethod
    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """Compute a parameter update with the same PyTree structure."""
        ...


class Layer(AbstractModule, ABC):
    """A trainable layer with an activation and reducer."""

    @property
    def has_state(self) -> bool:
        """Return True; layers are stateful by design."""
        return True

    @abstractmethod
    def activation(self, x: Array) -> Array:
        """Apply the layer’s activation function."""
        ...

    @abstractmethod
    def reduce(self, h: PyTree) -> Array:
        """Aggregate incoming messages into a single tensor."""
        ...


class Adapter(AbstractModule, ABC):
    """A stateless mapping between layers."""

    @property
    def has_state(self) -> bool:
        """Return False; adapters carry no persistent state."""
        return False
