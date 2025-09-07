from abc import abstractmethod, abstractproperty
from jax.typing import Array, PyTree
import equinox as eqx


class AbstractModule(eqx.Module):
    """abstract class for layers and adapters"""

    @abstractproperty
    def has_state(self) -> bool:
        pass

    @abstractmethod
    def __call__(self, x: Array) -> Array:
        pass

    @abstractmethod
    def activation(self, x: Array) -> Array:
        pass

    @abstractmethod
    def backward(self, s: Array, h: Array) -> "AbstractModule":
        pass


class Layer(AbstractModule):
    """specialization of Module. Symbolizes a
    a Layer."""

    def has_state(self) -> bool:
        return True

    @abstractmethod
    def reduce(self, h: PyTree[Array]) -> Array:
        pass


class Adapter(AbstractModule):
    """Adapters simply carry information from
    one layer to another"""

    def has_state(self) -> bool:
        return False
