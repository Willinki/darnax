from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar, cast

import jax.numpy as jnp
import numpy as np
from equinox import filter_jit
from jax import Array

from darnax.orchestrators.interface import AbstractOrchestrator
from darnax.states.interface import State

StateT = TypeVar("StateT", bound=State)
OrchestratorT = TypeVar("OrchestratorT", bound=AbstractOrchestrator[Any])
Ctx = dict[str, Any]

TrainCore = Callable[
    [Array, Array, Array, OrchestratorT, StateT, dict[str, Any]],
    tuple[Array, OrchestratorT, StateT, Ctx],
]
EvalCore = Callable[
    [Array, Array, Array, OrchestratorT, StateT, dict[str, Any]],
    tuple[Array, StateT, Mapping[str, Array]],
]


class Trainer(ABC, Generic[OrchestratorT, StateT]):
    r"""Two-step trainer with a single context dictionary.

    This class exposes a minimal, stateful API (``train_step``, ``eval_step``)
    wrapping pure, JIT-compiled static methods (``_train_step_impl``, ``_eval_step_impl``).
    The class mutates its attributes by rebinding after each pure call.

    Parameters
    ----------
    orchestrator : AbstractOrchestrator
        The model/orchestrator object. Treated as read-only inside eval by convention.
    state : State
        A mutable state pytree used by the step functions (e.g., buffers, EMA, counters).
    ctx : dict[str, Any]
        A single "bag" of step requirements:
        - Required:
            - ``"optimizer"`` : object with update semantics (e.g., optax transform)
            - ``"opt_state"`` : optimizer state pytree
        - Optional/common (conventions):
            - ``"t"`` : global step counter (int or JAX scalar). If provided as int, it is
              normalized to a JAX scalar once in :meth:`validate_ctx`.
            - Any other immutable/static knobs (callables, Python scalars) that **do not change**
              identity across steps.
            - Any mutable arrays/pytree values needed by the steps (e.g., schedulers' internal
              state) should be JAX arrays so they can change without recompilation.

    Notes
    -----
    - ``ctx`` must keep a **stable set of keys** to avoid retraces.
    - Non-array objects in ``ctx`` are treated as **static** under ``filter_jit``; changing
      their identity causes recompilation.
    - Arrays (and pytrees of arrays) in ``ctx`` may change freely across steps without
      triggering recompilation.

    Attributes
    ----------
    orchestrator : AbstractOrchestrator
    state : State
    ctx : dict[str, Any]

    """

    orchestrator: OrchestratorT
    state: StateT
    ctx: Ctx

    def __init__(self) -> None:
        """Initialize the jittable cores as None."""
        self._jit_train: TrainCore[OrchestratorT, StateT] | None = None
        self._jit_eval: EvalCore[OrchestratorT, StateT] | None = None

    def train_step(
        self,
        x: Array,
        y: Array,
        rng: Array,
        use_gating: bool = False,
        gating_shift: float = 1.0,
        fake_dynamics: bool = False,
        fake_dynamics_k: float = 0.25,
        fake_dynamics_vanilla: bool = True,
        double_dynamics: bool = True,
    ) -> Array:
        r"""Run one training step.

        Calls the pure, JIT-compiled :meth:`_train_step_impl`, then rebinds
        ``orchestrator``, ``state``, and ``ctx`` with the returned values.

        Parameters
        ----------
        x : Array
            Input batch.
        y : Array
            Target labels.
        rng : Array
            Random key.
        use_gating : bool
            If true, we apply gating to the rule with a global scalar.

        Returns
        -------
        Array
            Updated random key.

        Notes
        -----
        This method may update:
        - ``orchestrator`` (model parameters),
        - ``state`` (mutable buffers, EMA, etc.),
        - ``ctx`` (e.g., ``"opt_state"``, schedulers, counters).

        """
        if self._jit_train is None:
            core = type(self)._train_step_impl
            self._jit_train = cast("TrainCore[OrchestratorT, StateT]", filter_jit(core))

        rng, orch, st, ctx, logs = self._jit_train(
            x,
            y,
            rng,
            self.orchestrator,
            self.state,
            self.ctx,
            use_gating=use_gating,
            gating_shift=gating_shift,
            fake_dynamics=fake_dynamics,
            fake_dynamics_k=fake_dynamics_k,
            fake_dynamics_vanilla=fake_dynamics_vanilla,
            double_dynamics=double_dynamics,
        )

        fraction_updated_win = (
            np.abs(self.orchestrator.lmap[1][0].W - orch.lmap[1][0].W) > 0.0001
        ).mean()
        fraction_updated_J = (
            np.abs(self.orchestrator.lmap[1][1].J - orch.lmap[1][1].J) > 0.0001
        ).mean()
        fraction_updated_wout = (
            np.abs(self.orchestrator.lmap[2][1].W - orch.lmap[2][1].W) > 0.0001
        ).mean()
        fraction_updated_wback = (
            np.abs(self.orchestrator.lmap[1][2].W - orch.lmap[1][2].W) > 0.0001
        ).mean()

        avg_magnitude_win = np.abs(orch.lmap[1][0].W).mean()
        avg_magnitude_J = np.abs(orch.lmap[1][1].J).mean()
        avg_magnitude_wout = np.abs(orch.lmap[2][1].W).mean()
        avg_magnitude_wback = np.abs(orch.lmap[1][2].W).mean()
        logs.update(
            {
                "fraction_updated/W_in": fraction_updated_win,
                "fraction_updated/J": fraction_updated_J,
                "fraction_updated/W_out": fraction_updated_wout,
                "fraction_updated/W_back": fraction_updated_wback,
                "avg_magnitude/W_in": avg_magnitude_win,
                "avg_magnitude/J": avg_magnitude_J,
                "avg_magnitude/W_out": avg_magnitude_wout,
                "avg_magnitude/W_back": avg_magnitude_wback,
            }
        )

        self.orchestrator, self.state, self.ctx = orch, st, ctx
        return rng, logs

    def eval_step(self, x: Array, y: Array, rng: Array) -> tuple[Array, Mapping[str, Array]]:
        r"""Evaluate on one batch.

        Calls the pure, JIT-compiled :meth:`_eval_step_impl`, then rebinds ``state``.
        Returns a mapping of metrics that are **not** stored in :attr:`ctx`.

        Parameters
        ----------
        x : Array
            Input batch.
        y : Array
            Target labels.
        rng : Array
            Random key.

        Returns
        -------
        rng : Array
            Updated random key.
        metrics : Mapping[str, Array]
            Dictionary of metric arrays computed on this batch.

        Notes
        -----
        By convention, evaluation treats the orchestrator and static pieces as read-only.
        Only ``state`` may be updated (e.g., running statistics).

        """
        if self._jit_eval is None:
            core = type(self)._eval_step_impl
            self._jit_eval = cast("EvalCore[OrchestratorT, StateT]", filter_jit(core))

        rng, st, metrics = self._jit_eval(x, y, rng, self.orchestrator, self.state, self.ctx)
        self.state = st
        return rng, metrics

    # -------------------- Pure cores --------------------

    @staticmethod
    @abstractmethod
    def _train_step_impl(
        x: Array,
        y: Array,
        rng: Array,
        orchestrator: OrchestratorT,
        state: StateT,
        ctx: dict[str, Any],
        use_gating: bool,
    ) -> tuple[Array, OrchestratorT, StateT, Ctx]:
        r"""Pure training step to be JIT-compiled.

        This function must be **pure**: it receives pytrees and returns new pytrees
        without side effects. The outer :meth:`train_step` handles in-place rebinding.

        Parameters
        ----------
        x : Array
            Input batch.
        y : Array
            Target labels.
        rng : Array
            Random key.
        orchestrator : AbstractOrchestrator
            Current model/orchestrator.
        state : State
            Current mutable state.
        ctx : dict[str, Any]
            Context dict with optimizer, opt_state, counters, and any other requirements.
        use_gating: bool.
            If true we apply a scalar gate to the update rule

        Returns
        -------
        rng : Array
            Updated random key.
        orchestrator : AbstractOrchestrator
            Updated orchestrator.
        state : State
            Updated state.
        ctx : dict[str, Any]
            Updated context dictionary.

        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _eval_step_impl(
        x: Array,
        y: Array,
        rng: Array,
        orchestrator: OrchestratorT,
        state: StateT,
        ctx: dict[str, Any],
    ) -> tuple[Array, StateT, Mapping[str, Array]]:
        r"""Pure evaluation step to be JIT-compiled.

        This function must be **pure** and should not modify static objects.
        It may update ``state`` (e.g., running statistics).

        Parameters
        ----------
        x : Array
            Input batch.
        y : Array
            Target labels.
        rng : Array
            Random key.
        orchestrator : AbstractOrchestrator
            Current model/orchestrator (read-only by convention).
        state : State
            Current mutable state.
        ctx : dict[str, Any]
            Context dict (read-only by convention during eval).

        Returns
        -------
        rng : Array
            Updated random key.
        state : State
            Updated state.
        metrics : Mapping[str, Array]
            Dictionary of evaluation metrics.

        """
        raise NotImplementedError

    # -------------------- One-time validation --------------------

    @staticmethod
    def validate_ctx(ctx: Ctx) -> Ctx:
        r"""Validate and normalize the context dictionary.

        Called once in ``__init__`` to fail fast and to normalize counters.

        Parameters
        ----------
        ctx : dict[str, Any]
            User-provided context dictionary.

        Returns
        -------
        dict[str, Any]
            A normalized copy of the context dictionary.

        Raises
        ------
        ValueError
            If required keys are missing.

        Notes
        -----
        - Ensures that ``"optimizer"`` and ``"opt_state"`` are present.
        - Converts ``ctx["t"]`` to a JAX scalar if it is a Python integer.
        - Returns a plain ``dict`` to keep structure explicit and stable.

        """
        for k in ("optimizer", "opt_state"):
            if k not in ctx:
                raise ValueError(f"ctx missing required key '{k}'")

        out = dict(ctx)
        if "t" in out and not hasattr(out["t"], "dtype"):
            out["t"] = jnp.asarray(out["t"], dtype=jnp.int32)
        return out
