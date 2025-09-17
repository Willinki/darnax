from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import equinox as eqx
import jax

from bionet.layer_maps.sparse import LayerMap

if TYPE_CHECKING:
    from bionet.modules.interfaces import AbstractModule
    from bionet.states.interface import State

    ModuleT = TypeVar("ModuleT", bound="AbstractModule")

StateT = TypeVar("StateT", bound="State")
KeyArray = jax.Array


class AbstractOrchestrator(eqx.Module, Generic[StateT]):
    """Handles communication and messages between modules in a layermap."""

    lmap: LayerMap

    @abstractmethod
    def step(self, state: StateT, *, rng: jax.Array) -> tuple[StateT, KeyArray]:
        """Given the current state, run one forward/update step and return the new state."""
        pass

    @abstractmethod
    def step_inference(self, state: StateT, *, rng: jax.Array) -> tuple[StateT, KeyArray]:
        """Given the current state, run one forward/update step and return the new state.

        Does not compute messages traveling "to the right".
        """
        pass

    @abstractmethod
    def backward(self, state: StateT, rng: KeyArray) -> LayerMap:
        """Given the current state, compute the updates for the parameters in `lmap`.

        The returned PyTree must have the same structure as `lmap` so that it can
        be used with Optax's `update` and `apply_updates`.
        """
        pass
