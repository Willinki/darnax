from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from jax import Array

from bionet.layer_maps.sparse import LayerMap
from bionet.orchestrators.interface import AbstractOrchestrator
from bionet.states.sequential import SequentialState

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bionet.modules.interfaces import AbstractModule

    KeyArray = Array


class SequentialOrchestrator(AbstractOrchestrator[SequentialState]):
    """Sequential message-passing orchestrator compatible with the new LayerMap.

    Assumptions
    -----------
    - `lmap` is a dict-of-dicts PyTree with **static structure** (sorted keys) and
      Equinox modules as values; the LayerMap flattens *through* modules so their
      parameters are visible to JAX/Optax.
    - For each receiver `i`, the diagonal module `lmap[i, i]` implements
      `reduce(pytree_of_messages)` and `activation(Array) -> Array`.
    - Each edge module `lmap[i, j]` is callable as `module(x, rng=...) -> Array`
      and provides `backward(x, y, y_hat) -> AbstractModule` (same PyTree type).
    """

    lmap: LayerMap  # not static; module parameters will be updated externally

    def __init__(self, layers: LayerMap):
        """Initialize the orchestrator from the layermap."""
        self.lmap = layers

    # ---------------------------- public API ----------------------------

    def step(
        self,
        state: SequentialState,
        *,
        rng: KeyArray,
    ) -> tuple[SequentialState, KeyArray]:
        """One forward update over all receivers.

        Parameters
        ----------
        state : SequentialState
            The current network state.
        rng : KeyArray
            PRNG key; will be split per receiver/sender and advanced.

        """
        for receiver_idx, senders_group in self.lmap.row_items():
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
        """One forward update over all receivers.

        Parameters
        ----------
        state : SequentialState
            The current network state.
        rng : KeyArray
            PRNG key; will be split per receiver/sender and advanced.

        """
        for receiver_idx, senders_group in self.lmap.row_items():
            filtered_senders_group = {k: v for k, v in senders_group.items() if k >= receiver_idx}
            rng, sub = jax.random.split(rng)
            messages = self._compute_messages(filtered_senders_group, state, rng=sub)
            aggregated: Array = self.lmap[receiver_idx, receiver_idx].reduce(messages)
            activated: Array = self.lmap[receiver_idx, receiver_idx].activation(aggregated)
            state = state.replace_val(receiver_idx, activated)
        return state, rng

    def backward(self, state: SequentialState, rng: KeyArray) -> LayerMap:
        """Compute per-edge updates for all modules.

        Returns a LayerMap-structured pytree of updates (same PyTree type as modules).
        """
        # First pass: per-receiver messages and aggregated activation kept under key `i`.
        activations: dict[int, dict[int, Array]] = {}
        for receiver_idx, senders_group in self.lmap.row_items():
            rng, sub = jax.random.split(rng)
            msgs = self._compute_messages(senders_group, state, rng=sub)
            # Add the receiver's aggregated activation under its own key.
            msgs[receiver_idx] = self.lmap[receiver_idx, receiver_idx].reduce(
                {k: v for k, v in msgs.items() if k <= receiver_idx}
            )  # IMPORTANT: in the backward we dont consider messages from the right
            activations[receiver_idx] = msgs
        # Second pass: ask each module for its update.
        return self._backward_direct(state, activations)

    # ---------------------------- internals ----------------------------

    def _backward_direct(
        self,
        state: SequentialState,
        activations: dict[int, dict[int, Array]],
    ) -> LayerMap:
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
        """Compute messages from all senders in `layer_group` to its receiver.

        The `layer_group` is a read-only mapping (from LayerMap) of `sender_idx -> module`.
        """
        messages: dict[int, Array] = {}
        for sender_idx, module in layer_group.items():
            rng, sub = jax.random.split(rng)
            messages[sender_idx] = module(state[sender_idx], rng=sub)
        return messages
