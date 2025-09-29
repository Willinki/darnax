"""Sequential orchestrator for layered networks.

This module provides :class:`SequentialOrchestrator`, a concrete implementation
of the orchestrator contract that traverses a :class:`~darnax.layer_maps.sparse.LayerMap`
row-by-row (deterministic order), computes messages along edges, reduces them
at each receiver, and updates a :class:`~darnax.states.sequential.SequentialState`.

Overview
--------
- :meth:`step` updates **all receivers except the output row** (keeps ``state[-1]`` unchanged).
- :meth:`step_inference` is like :meth:`step` but **skips right-going edges**
  (uses a forward-only schedule; cheaper partial refresh).
- :meth:`predict` computes the **output buffer** from current internal buffers.
- :meth:`backward` builds a **LayerMap-shaped PyTree of parameter updates** by
  calling each module’s local rule. The method returns a new orchestrator whose
  ``lmap`` contains *updates* (not new parameters), so it can be fed directly
  to Optax’s ``apply_updates``.

RNG handling
------------
All public methods accept an RNG key and return a **new** key. Internally,
keys are split **per receiver** and **per sender** to keep randomness
reproducible and side-effect free.

See Also
--------
tutorials/05_orchestrators.md
tutorials/06_simple_net_on_artificial_data.md

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from jax import Array

from darnax.layer_maps.sparse import LayerMap
from darnax.orchestrators.interface import AbstractOrchestrator
from darnax.states.sequential import SequentialState

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    from darnax.modules.interfaces import AbstractModule

    KeyArray = Array


class SequentialOrchestrator(AbstractOrchestrator[SequentialState]):
    """Sequential message-passing orchestrator.

    Assumptions
    -----------
    - ``lmap`` has **static structure** (sorted integer keys); values are Equinox
      modules that are PyTrees (parameters visible to JAX/Optax).
    - For each receiver ``i``, the diagonal module ``lmap[i, i]`` implements
      ``reduce(pytree_of_messages)`` and ``activation(Array) -> Array``.
    - Each edge module ``lmap[i, j]`` is callable as ``module(x, rng=...) -> Array``
      and exposes ``backward(x, y, y_hat) -> AbstractModule`` (same PyTree type).

    Notes
    -----
    This orchestrator is **pure**: given a state and RNG, it returns a new state
    and a new RNG key. All scheduling choices (row order, forward-only mode) are
    explicit and deterministic.

    """

    lmap: LayerMap  # not static; module parameters will be updated externally

    def __init__(self, layers: LayerMap):
        """Initialize the orchestrator from a layermap.

        Parameters
        ----------
        layers : LayerMap
            Static adjacency (rows/cols) with Equinox modules as values.

        """
        self.lmap = layers

    # ---------------------------- public API ----------------------------

    def step(
        self,
        state: SequentialState,
        *,
        rng: KeyArray,
    ) -> tuple[SequentialState, KeyArray]:
        """Run one forward/update sweep for all receivers **except output**.

        For each receiver row ``i`` (excluding the last), the orchestrator:
        (1) computes messages from all neighbors ``j`` in ``lmap[i]``,
        (2) reduces them via ``lmap[i, i].reduce(messages)``,
        (3) applies ``lmap[i, i].activation(...)``,
        (4) writes the result into ``state[i]``.

        Parameters
        ----------
        state : SequentialState
            Current global state.
        rng : KeyArray
            PRNG key (split per receiver/sender).

        Returns
        -------
        (new_state, new_rng) : tuple[SequentialState, KeyArray]
            Updated state (output unchanged) and an advanced RNG key.

        """
        # we avoid computing the update of the last state component,
        # since it is clamped.
        for receiver_idx, senders_group in self.lmap.row_items(skip_last=True):
            rng, sub = jax.random.split(rng)
            messages = self._compute_messages(senders_group, state, rng=sub)
            aggregated: Array = self.lmap[receiver_idx, receiver_idx].reduce(messages)
            activated: Array = self.lmap[receiver_idx, receiver_idx].activation(aggregated)
            state = state.replace_val(receiver_idx, activated)
        return state, rng

    def step_inference(
        self,
        state: SequentialState,
        *,
        rng: KeyArray,
    ) -> tuple[SequentialState, KeyArray]:
        """Run a **forward-only** sweep (skip output and right-going edges).

        Like :meth:`step`, but calls ``row_items(skip_last=True, forward_only=True)``,
        which filters neighbor edges to a forward schedule (commonly ``j <= i``).

        Parameters
        ----------
        state : SequentialState
            Current global state.
        rng : KeyArray
            PRNG key (split per receiver/sender).

        Returns
        -------
        (new_state, new_rng) : tuple[SequentialState, KeyArray]
            Updated state (output unchanged) and an advanced RNG key.

        """
        # we only consider forward messages, not backwards
        # and we also skip the last element
        for receiver_idx, senders_group in self.lmap.row_items(skip_last=True, forward_only=True):
            rng, sub = jax.random.split(rng)
            messages = self._compute_messages(senders_group, state, rng=sub)
            aggregated: Array = self.lmap[receiver_idx, receiver_idx].reduce(messages)
            activated: Array = self.lmap[receiver_idx, receiver_idx].activation(aggregated)
            state = state.replace_val(receiver_idx, activated)
        return state, rng

    def predict(self, state: SequentialState, rng: KeyArray) -> tuple[SequentialState, KeyArray]:
        """Compute/refresh the **output** buffer ``state[-1]`` from current buffers.

        Parameters
        ----------
        state : SequentialState
            Current global state (uses existing upstream buffers).
        rng : KeyArray
            PRNG key (split internally).

        Returns
        -------
        (new_state, new_rng) : tuple[SequentialState, KeyArray]
            A state where the output buffer has been updated, and a fresh RNG key.

        """
        receiver_idx = self.lmap.rows()[-1]
        senders_group = self.lmap.neighbors(receiver_idx)
        rng, sub = jax.random.split(rng)
        messages = self._compute_messages(senders_group, state, rng=sub)
        aggregated: Array = self.lmap[receiver_idx, receiver_idx].reduce(messages)
        activated: Array = self.lmap[receiver_idx, receiver_idx].activation(aggregated)
        state = state.replace_val(-1, activated)
        return state, rng

    def backward(self, state: SequentialState, rng: KeyArray) -> Self:
        """Compute per-edge parameter updates and return them as an orchestrator.

        Two-phase algorithm:

        1) **Activation pass.** For each receiver ``i`` (including the output),
           compute messages ``msg[j] = lmap[i, j](state[j], rng=...)`` from
           all neighbors, then compute the receiver’s aggregate **using forward
           messages only** (e.g., ``j <= i``) and store it under ``msg[i]``.
        2) **Local rules.** For every edge ``(i, j)``, call
           ``lmap[i, j].backward(x=state[j], y=state[i], y_hat=msg[j])`` and
           assemble the results into a new LayerMap with the **same key layout**.

        The method returns ``type(self)(layers=updates)`` where ``updates`` is
        the LayerMap of per-edge updates. This allows ``optax.apply_updates`` to
        be used directly with the returned PyTree.

        Parameters
        ----------
        state : SequentialState
            Current global state (provides inputs and targets for local rules).
        rng : KeyArray
            PRNG key (split per receiver/sender).

        Returns
        -------
        Self
            An orchestrator whose ``lmap`` contains **updates** (not new params),
            mirroring the original layermap structure.

        Notes
        -----
        The receiver aggregate used for learning is computed with **forward-only**
        messages (no right-going edges), matching the schedule used in inference.

        """
        activations: dict[int, dict[int, Array]] = {}
        # here we compute all activations (back, forth, and last)
        for receiver_idx, senders_group in self.lmap.row_items():
            rng, sub = jax.random.split(rng)
            msgs = self._compute_messages(senders_group, state, rng=sub)
            # Add the receiver's aggregated activation under its own key.
            msgs[receiver_idx] = self.lmap[receiver_idx, receiver_idx].reduce(
                {k: v for k, v in msgs.items() if k <= receiver_idx}
            )  # IMPORTANT: in the backward we dont consider backward messages when aggregating
            activations[receiver_idx] = msgs
        # Second pass: ask each module for its update.
        return type(self)(layers=self._backward_direct(state, activations))

    # ---------------------------- internals ----------------------------

    def _backward_direct(
        self,
        state: SequentialState,
        activations: dict[int, dict[int, Array]],
    ) -> LayerMap:
        """Assemble a LayerMap of per-edge updates from local rules.

        Parameters
        ----------
        state : SequentialState
            Current global state (provides ``x=state[j]`` and ``y=state[i]``).
        activations : dict[int, dict[int, Array]]
            For each receiver ``i``, a mapping ``j -> y_hat_ij`` containing
            edge predictions/messages (and the receiver aggregate under key ``i``).

        Returns
        -------
        LayerMap
            A LayerMap with the same key layout as ``self.lmap`` whose values
            are per-module **updates** (PyTrees shaped like the original modules).

        """
        updates: dict[int, dict[int, AbstractModule]] = {}
        for receiver_idx, sent_messages in activations.items():
            rec_updates: dict[int, AbstractModule] = {}
            for sender_idx, message in sent_messages.items():
                rec_updates[sender_idx] = self.lmap[receiver_idx, sender_idx].backward(
                    x=state[sender_idx],
                    y=state[receiver_idx],
                    y_hat=message,
                )
            updates[receiver_idx] = rec_updates
        return LayerMap.from_dict(updates, require_diagonal=True)

    def _compute_messages(
        self,
        layer_group: Mapping[int, AbstractModule],
        state: SequentialState,
        *,
        rng: KeyArray,
    ) -> dict[int, Array]:
        """Compute messages from all senders in ``layer_group`` to its receiver.

        Parameters
        ----------
        layer_group : Mapping[int, AbstractModule]
            Read-only mapping (from :class:`LayerMap`) of ``sender_idx -> module``.
        state : SequentialState
            Global state providing inputs ``state[sender_idx]`` for each sender.
        rng : KeyArray
            PRNG key to split per sender.

        Returns
        -------
        dict[int, Array]
            A dictionary mapping ``sender_idx`` to the computed message
            ``module(state[sender_idx], rng=subkey)``.

        """
        messages: dict[int, Array] = {}
        for sender_idx, module in layer_group.items():
            rng, sub = jax.random.split(rng)
            messages[sender_idx] = module(state[sender_idx], rng=sub)
        return messages
