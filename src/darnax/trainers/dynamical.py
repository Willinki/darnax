from collections.abc import Mapping
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from optax import GradientTransformation

from darnax.orchestrators.interface import AbstractOrchestrator
from darnax.states.interface import State
from darnax.trainers.interface import Trainer
from darnax.trainers.utils import batch_accuracy, scan_n
from darnax.utils.typing import PyTree

StateT = TypeVar("StateT", bound=State)
OrchestratorT = TypeVar("OrchestratorT", bound=AbstractOrchestrator[Any])
Ctx = dict[str, Any]


class DynamicalTrainer(Trainer[OrchestratorT, StateT], Generic[OrchestratorT, StateT]):
    """Minimal trainer that implements the learning rule described in the main paper.

    This trainer wires together an orchestrator and a per-batch `State`,
    performing warmup/clamped/free dynamics for training. Warmup/free
    for evaluation. It assumes a local learning rule implemented by the
    orchestrator and an Optax optimizer for parameter updates.

    More specifically:
    - Warmup: Initial dynamical phase (1 or 2 iterations) with only left fields
    - Clamped: Dynamical phase with both left and right fields
    - Free: Final dynamical phase where only the left fields are used

    During evaluation only warmup+free is performed.

    Parameters
    ----------
    orchestrator : AbstractOrchestrator[StateT]
        The model/orchestrator. Treated as read-only during eval by convention.
    state : StateT
        Mutable per-batch state pytree (buffers, EMA, etc.).
    optimizer : GradientTransformation
        Optax transform (e.g., `optax.adam(...)`).
    optimizer_state : PyTree
        Optax optimizer state (from `optimizer.init(params)`).
    warmup_n_iter : int, default=1
        Number of warmup iterations per batch.
    train_clamped_n_iter : int, default=7
        Number of clamped iterations per batch (train only).
    train_free_n_iter : int, default=7
        Number of free iterations per batch (train only).
    eval_n_iter : int, default=14
        Number of free iterations per batch (eval).

    Attributes
    ----------
    orchestrator : OrchestratorT
        The current orchestrator/model.
    state : StateT
        The current mutable state.
    ctx : dict[str, Any]
        A context dictionary with optimizer, optimizer state, and iteration counts.
        Keys are:
        - "optimizer" : GradientTransformation
        - "optimizer_state" : PyTree
        - "warmup_iter" : int
        - "clamped_iter" : int
        - "free_iter" : int
        - "eval_iter" : int

    """

    orchestrator: OrchestratorT
    state: StateT
    ctx: Ctx

    def __init__(
        self,
        orchestrator: OrchestratorT,
        state: StateT,
        optimizer: GradientTransformation,
        optimizer_state: PyTree,
        warmup_n_iter: int = 1,
        train_clamped_n_iter: int = 7,
        train_free_n_iter: int = 7,
        eval_n_iter: int = 14,
    ) -> None:
        """Initialize dynamical trainer."""
        super().__init__()
        self.orchestrator = orchestrator
        self.state = state
        self.ctx = {
            "optimizer": optimizer,
            "optimizer_state": optimizer_state,
            "warmup_iter": warmup_n_iter,
            "clamped_iter": train_clamped_n_iter,
            "free_iter": train_free_n_iter,
            "eval_iter": eval_n_iter,
        }
        self.ctx = self.validate_ctx(self.ctx)

    @staticmethod
    def _train_step_impl(
        x: Array,
        y: Array,
        rng: Array,
        orchestrator: OrchestratorT,
        state: StateT,
        ctx: dict[str, Any],
    ) -> tuple[Array, OrchestratorT, StateT, Ctx]:
        """Pure training core: warmup → clamped → free, then optimizer step.

        Parameters
        ----------
        x : Array
            Input batch.
        y : Array
            Target labels.
        rng : Array
            RNG key.
        orchestrator : OrchestratorT
            Current orchestrator/model.
        state : StateT
            Current per-batch state.
        ctx : Mapping[str, Any]
            Read-only context (optimizer, optimizer_state, iteration counts).

        Returns
        -------
        rng : Array
            Updated RNG key.
        orchestrator : OrchestratorT
            Updated orchestrator after parameter updates.
        state : StateT
            Updated state after rollouts.
        ctx : dict[str, Any]
            New context with updated `"optimizer_state"`; other keys unchanged.

        Notes
        -----
        This function must be pure; no in-place mutation of `ctx` or object attributes.

        """
        # 1) per-batch init
        state = state.init(x, y)

        # 2) rollout phases
        (state, rng), _ = scan_n(
            orchestrator.step, (state, rng), n_iter=ctx["warmup_iter"], filter_messages="forward"
        )
        (state, rng), _ = scan_n(
            orchestrator.step, (state, rng), n_iter=ctx["clamped_iter"], filter_messages="all"
        )
        (state, rng), _ = scan_n(
            orchestrator.step, (state, rng), n_iter=ctx["free_iter"], filter_messages="forward"
        )

        # 3) local/backprop deltas shaped like orchestrator
        grads = orchestrator.backward(state, rng=rng)

        # 4) filter trainable leaves
        params = eqx.filter(orchestrator, eqx.is_inexact_array)
        grads = eqx.filter(grads, eqx.is_inexact_array)

        # 5) optimizer step (consistent key: "optimizer_state")
        updates, new_opt_state = ctx["optimizer"].update(
            grads, ctx["optimizer_state"], params=params
        )
        new_orch = eqx.apply_updates(orchestrator, updates)

        # 6) return a NEW ctx (pure)
        new_ctx: Ctx = eqx.tree_at(lambda d: d["optimizer_state"], ctx, new_opt_state)
        return rng, new_orch, state, new_ctx

    @staticmethod
    def _eval_step_impl(
        x: Array,
        y: Array,
        rng: Array,
        orchestrator: OrchestratorT,
        state: StateT,
        ctx: dict[str, Any],
    ) -> tuple[Array, StateT, Mapping[str, Array]]:
        """Pure evaluation core: warmup → free, then predict and compute metrics.

        Parameters
        ----------
        x : Array
            Input batch.
        y : Array
            Target labels.
        rng : Array
            RNG key.
        orchestrator : OrchestratorT
            Current orchestrator/model (read-only by convention).
        state : StateT
            Current per-batch state.
        ctx : Mapping[str, Any]
            Read-only context with iteration counts.

        Returns
        -------
        rng : Array
            Updated RNG key.
        state : StateT
            Updated state after rollouts and prediction.
        metrics : Mapping[str, Array]
            Dictionary of evaluation metrics, at least ``{"accuracy": scalar}``.

        Notes
        -----
        This function should not modify static objects; only `state` is updated.

        """
        # 1) per-batch init
        state = state.init(x, y)

        # 2) warmup + free eval dynamics
        (state, rng), _ = scan_n(
            orchestrator.step, (state, rng), n_iter=ctx["warmup_iter"], filter_messages="forward"
        )
        (state, rng), _ = scan_n(
            orchestrator.step, (state, rng), n_iter=ctx["eval_iter"], filter_messages="forward"
        )

        # 3) prediction
        state, rng = orchestrator.predict(state, rng=rng)
        y_pred = state.readout

        # 4) metrics
        accuracy = batch_accuracy(y_true=jnp.asarray(y), y_pred=jnp.asarray(y_pred))
        return rng, state, {"accuracy": accuracy}

    @staticmethod
    def validate_ctx(ctx: Ctx) -> Ctx:
        """Validate and normalize the context dictionary.

        Ensures required keys are present and returns a normalized copy.

        Parameters
        ----------
        ctx : dict[str, Any]
            Context dictionary supplied at construction.

        Returns
        -------
        dict[str, Any]
            Normalized context dictionary.

        Raises
        ------
        ValueError
            If any required key is missing.

        """
        required_keys = [
            "optimizer",
            "optimizer_state",
            "warmup_iter",
            "clamped_iter",
            "free_iter",
            "eval_iter",
        ]
        for key in required_keys:
            if key not in ctx:
                raise ValueError(f"ctx missing required key '{key}'")
        return dict(ctx)
