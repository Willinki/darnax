from abc import abstractmethod
import equinox as eqx
from jax.typing import PyTree
from src.utils.layer_map import LayerMap
from src.states.interface import State


class AbstractOrchestrator(eqx.Module):
    lmap: LayerMap
    _inference: bool = eqx.field(static=True, default=False)

    @property
    def inference(self) -> bool:
        """Whether orchestrator is in inference mode."""
        return self._inference

    @inference.setter
    def inference(self, value: bool) -> None:
        object.__setattr__(self, "_inference", bool(value))

    @abstractmethod
    def step(self, state: State) -> State:
        """
        Given the current state, run one forward/update step and
        return the new state.
        """
        pass

    @abstractmethod
    def backward(self, state: State) -> PyTree:
        """
        Given the current state, compute the updates for the parameters in `lmap`.
        The returned PyTree must have the same structure as `lmap` so that it can
        be used with Optax's `update` and `apply_updates`.
        """
        pass
