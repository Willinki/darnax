from abc import abstractmethod
from typing import Any, Self

import equinox as eqx
from jax import Array

from darnax.utils.typing import PyTree


class State(eqx.Module):
    """Global state for a network (layermap).

    Each layer of a layermap has an associated state.
    The number of elements in the global state needs to be
    equal to the number of layers in the LayerMap + 1 at least.

    See "inputs" and "output" in docs.
    """

    @abstractmethod
    def __getitem__(self, key: Any) -> Array:
        """Return an array for the given key."""

    @abstractmethod
    def init(self, x: Array, y: Array | None = None) -> Self:
        """Initialize the state with the input and output in the correct place.

        They are the first [0] and last element of the state.
        """

    @abstractmethod
    def replace(self, value: PyTree) -> Self:
        """Replace the whole state functionally with new values."""

    @abstractmethod
    def replace_val(self, idx: Any, value: Array) -> Self:
        """Return a new instance with layer state at id idx replaced by value ."""
