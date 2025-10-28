"""Trainer interface with stateful API and pure functional core."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from equinox import filter_jit

from darnax.states.interface import State

if TYPE_CHECKING:
    from jax import Array

    from darnax.orchestrators.interface import AbstractOrchestrator

StateT = TypeVar("StateT", bound=State)
OrchestratorT = TypeVar("OrchestratorT", bound="AbstractOrchestrator[Any]")
ExtrasT = TypeVar("ExtrasT", bound=Mapping[str, Any])
ParamsT = TypeVar("ParamsT", bound=Mapping[str, Any])


class Trainer(ABC, Generic[StateT, OrchestratorT, ExtrasT, ParamsT]):
    """Stateful trainer wrapping a pure functional JAX core.

    Stateful methods mutate in-place, pure methods are JIT-compiled.
    """

    orchestrator: OrchestratorT
    state: StateT
    extras: ExtrasT
    params: ParamsT

    def train_step(
        self,
        x: Array,
        y: Array,
        rng: Array,
    ) -> Array:
        """Call the pure JIT-compiled training step, mutating trainer in-place.

        Parameters
        ----------
        x : Array
            Input batch
        y : Array
            Target labels
        rng : Array
            Random key

        Returns
        -------
        rng : Array
            Updated random key

        """
        jit_train_step_impl = filter_jit(self._train_step_impl)
        rng, self.orchestrator, self.state, self.extras = jit_train_step_impl(
            x, y, rng, self.orchestrator, self.state, self.extras, self.params
        )
        return rng

    def eval_step(
        self,
        x: Array,
        y: Array,
        rng: Array,
    ) -> tuple[Array, Array]:
        """Evaluate on one batch, mutating state only.

        Parameters
        ----------
        x : Array
            Input batch
        y : Array
            Target labels
        rng : Array
            Random key

        Returns
        -------
        rng : Array
            Updated random key
        accuracy : Array
            Evaluation accuracy

        """
        jit_eval_step_impl = filter_jit(self._eval_step_impl)
        rng, self.orchestrator, self.state, self.extras, accuracy = jit_eval_step_impl(
            x, y, rng, self.orchestrator, self.state, self.extras, self.params
        )
        return rng, accuracy

    @staticmethod
    @abstractmethod
    def _train_step_impl(
        x: Array,
        y: Array,
        rng: Array,
        orchestrator: OrchestratorT,
        state: StateT,
        extras: ExtrasT,
        params: ParamsT,
    ) -> tuple[Array, OrchestratorT, StateT, ExtrasT]:
        """Pure training step to be JIT-compiled.

        Should be a static method.

        Parameters
        ----------
        x : Array
            Input batch
        y : Array
            Target labels
        rng : Array
            Random key
        orchestrator : OrchestratorT
            Current model
        state : StateT
            Current state
        extras : ExtrasT
            Additional arguments to be updated (e.g., opt_state, optimizer)
        params : ParamsT
            Training parameters (e.g., t_train)

        Returns
        -------
        rng : Array
            Updated random key
        orchestrator : OrchestratorT
            Updated orchestrator
        state : StateT
            Updated state
        extras : ExtrasT
            Updated extras

        """
        pass

    @staticmethod
    @abstractmethod
    def _eval_step_impl(
        x: Array,
        y: Array,
        rng: Array,
        orchestrator: OrchestratorT,
        state: StateT,
        extras: ExtrasT,
        params: ParamsT,
    ) -> tuple[Array, OrchestratorT, StateT, ExtrasT, Array]:
        """Pure evaluation step to be JIT-compiled.

        Should be a static method.

        Parameters
        ----------
        x : Array
            Input batch
        y : Array
            Target labels
        rng : Array
            Random key
        orchestrator : OrchestratorT
            Current model (read-only)
        state : StateT
            Current state
        extras : ExtrasT
            Additional arguments to be updated (e.g., opt_state, optimizer)
        params : ParamsT
            Training parameters (e.g., t_train)

        Returns
        -------
        rng : Array
            Updated random key
        orchestrator : OrchestratorT
            Orchestrator (unchanged)
        state : StateT
            Updated state
        extras : ExtrasT
            Extras dict (unchanged)
        accuracy : Array
            Evaluation accuracy

        """
        pass
