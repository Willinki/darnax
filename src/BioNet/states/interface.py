import equinox as eqx
from jax.typing import Array
from abc import abstractmethod
from typing import Any, Optional


class State(eqx.Module):

    @abstractmethod
    def __getitem__(self, key: Any) -> Array:
        """Return an array (list) for the given key."""
        pass

    @abstractmethod
    def init(self, x: Array, y: Optional[Array] = None) -> "State":
        """"""
        pass
