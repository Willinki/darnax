"""Dynamical trainer implementing two-phase learning."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import equinox as eqx
import jax

from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.interface import Trainer

if TYPE_CHECKING:
    import optax
    from jax import Array


class DynamicalExtras(TypedDict):
    """Mutable training state for DynamicalTrainer."""

    optimizer: optax.GradientTransformation
    opt_state: optax.OptState


class DynamicalParams(TypedDict):
    """Immutable hyperparameters for DynamicalTrainer."""

    t_train: int
    t_eval: int


class DynamicalTrainer(
    Trainer[SequentialState, SequentialOrchestrator, DynamicalExtras, DynamicalParams]
):
    """Trainer for two-phase dynamical learning.

    Implements clamped + free dynamics with local plasticity.
    """

    def __init__(
        self,
        orchestrator: SequentialOrchestrator,
        state: SequentialState,
        optimizer: optax.GradientTransformation,
        t_train: int,
        t_eval: int,
    ):
        """Initialize dynamical trainer.

        Parameters
        ----------
        orchestrator : SequentialOrchestrator
            Network orchestrator with LayerMap topology
        state : SequentialState
            State with buffer sizes matching the network topology
        optimizer : optax.GradientTransformation
            Optax optimizer (e.g., optax.adam(2e-3))
        t_train : int, optional
            Dynamics steps for training
        t_eval : int, optional
            Dynamics steps for evaluation

        """
        self.orchestrator = orchestrator
        self.state = state

        opt_state = optimizer.init(eqx.filter(orchestrator, eqx.is_inexact_array))
        self.extras: DynamicalExtras = {"optimizer": optimizer, "opt_state": opt_state}
        self.params: DynamicalParams = {"t_train": t_train, "t_eval": t_eval}

    @staticmethod
    def _train_step_impl(
        x: Array,
        y: Array,
        rng: Array,
        orch: SequentialOrchestrator,
        s: SequentialState,
        extras: DynamicalExtras,
        params: DynamicalParams,
    ) -> tuple[Array, SequentialOrchestrator, SequentialState, DynamicalExtras]:
        """JIT-compiled training step with two-phase dynamics.

        Parameters
        ----------
        x : Array
            Input batch
        y : Array
            Target labels
        rng : Array
            Random key
        orch : SequentialOrchestrator
            Current orchestrator
        s : SequentialState
            Current state
        extras : DynamicalExtras
            Mutable training state (optimizer, opt_state)
        params : DynamicalParams
            Immutable hyperparameters (t_train, t_eval)

        Returns
        -------
        rng : Array
            Updated random key
        orch : SequentialOrchestrator
            Updated orchestrator
        s : SequentialState
            Updated state
        extras : DynamicalExtras
            Updated extras (new opt_state)

        """
        optimizer = extras["optimizer"]
        opt_state = extras["opt_state"]
        t_train = params["t_train"]

        # 1) Clamp inputs + labels into state
        s = s.init(x, y)

        # 2) Clamped phase: run full dynamics with labels
        def step_clamped(
            carry: tuple[SequentialState, Array], _: None
        ) -> tuple[tuple[SequentialState, Array], None]:
            state, key = carry
            state, key = orch.step(state, rng=key)
            return (state, key), None

        t_train_1 = t_train // 2
        (s, rng), _ = jax.lax.scan(step_clamped, (s, rng), xs=None, length=t_train_1)

        # 3) Free phase: run inference dynamics (no labels)
        def step_free(
            carry: tuple[SequentialState, Array], _: None
        ) -> tuple[tuple[SequentialState, Array], None]:
            state, key = carry
            state, key = orch.step_inference(state, rng=key)
            return (state, key), None

        t_train_2 = t_train - t_train_1
        (s, rng), _ = jax.lax.scan(step_free, (s, rng), xs=None, length=t_train_2)

        # 4) Compute local deltas + Optax update
        rng, update_key = jax.random.split(rng)
        grads = orch.backward(s, rng=update_key)

        # Filter to get only trainable params and grads
        filtered_orch = eqx.filter(orch, eqx.is_inexact_array)
        grads = eqx.filter(grads, eqx.is_inexact_array)

        # Apply optimizer update
        updates, opt_state = optimizer.update(grads, opt_state, params=filtered_orch)
        orch = eqx.apply_updates(orch, updates)

        # Update extras with new opt_state
        extras = {"optimizer": optimizer, "opt_state": opt_state}

        return rng, orch, s, extras

    @staticmethod
    def _eval_step_impl(
        x: Array,
        y: Array,
        rng: Array,
        orch: SequentialOrchestrator,
        s: SequentialState,
        extras: DynamicalExtras,
        params: DynamicalParams,
    ) -> tuple[Array, SequentialOrchestrator, SequentialState, DynamicalExtras, Array]:
        """JIT-compiled evaluation step with inference dynamics.

        Parameters
        ----------
        x : Array
            Input batch
        y : Array
            Target labels
        rng : Array
            Random key
        orch : SequentialOrchestrator
            Current orchestrator (read-only)
        s : SequentialState
            Current state
        extras : DynamicalExtras
            Evaluation extras (unused but maintained for signature)
        params : DynamicalParams
            Immutable hyperparameters (t_eval)

        Returns
        -------
        rng : Array
            Updated random key
        orch : SequentialOrchestrator
            Orchestrator (unchanged)
        s : SequentialState
            Updated state
        extras : DynamicalExtras
            Extras (unchanged)
        accuracy : Array
            Batch accuracy

        """
        t_eval = params["t_eval"]

        # 1) Clamp inputs only (no labels)
        s = s.init(x, None)

        # 2) Run inference dynamics
        def step_fn(
            carry: tuple[SequentialState, Array], _: None
        ) -> tuple[tuple[SequentialState, Array], None]:
            state, key = carry
            state, key = orch.step_inference(state, rng=key)
            return (state, key), None

        (s, rng), _ = jax.lax.scan(step_fn, (s, rng), xs=None, length=t_eval)

        # 3) Extract predictions
        s, rng = orch.predict(s, rng)
        y_hat = s[-1]  # Output is last element

        # 4) Compute accuracy
        predictions = jax.numpy.argmax(y_hat, axis=-1)
        targets = jax.numpy.argmax(y, axis=-1)
        accuracy = jax.numpy.mean(predictions == targets)

        return rng, orch, s, extras, accuracy
