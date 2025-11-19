"""Interfaces for Darnax modules.

This module defines the abstract contracts implemented by all computational
components in Darnax:

- ``AbstractModule`` is the common base for anything callable during the
  recurrent dynamics (layers and adapters).
- ``Layer`` is a **stateful** module that aggregates incoming messages and
  applies an activation.
- ``Adapter`` is a **stateless** mapping between layers (e.g., projections,
  reshapes, or wiring).

All classes are Equinox ``Module``s (i.e., PyTrees) and are compatible with
JAX transformations (``jit``, ``vmap``, ``grad``/**custom rules**, etc.).
"""

from abc import ABC, abstractmethod
from typing import Self

import equinox as eqx
from jax import Array

from darnax.utils.typing import PyTree

KeyArray = Array


class AbstractModule(eqx.Module, ABC):
    """Base class for layers and adapters.

    Subclasses must implement a pure functional forward pass and a local
    learning rule via :meth:`backward`. The object itself is a PyTree:
    parameter fields are leaves, and nested modules are subtrees. This makes
    instances compatible with Equinox/Optax update flows.

    Notes
    -----
    - The forward pass **must not** mutate parameters or hidden state.
      Any persistent state updates are orchestrated outside the call via the
      training loop (see tutorials).
    - ``rng`` is optional; when provided, it should be a JAX PRNG key
      (``KeyArray``). If ``None``, the module must behave deterministically.

    """

    @property
    @abstractmethod
    def has_state(self) -> bool:
        """Whether the module carries persistent state.

        Returns
        -------
        bool
            ``True`` if the module owns persistent state (e.g., activations,
            running stats, internal buffers) that is managed by the
            orchestrator/training loop; ``False`` otherwise.

        """
        ...

    @abstractmethod
    def __call__(self, x: Array, rng: Array | None = None) -> Array:
        """Compute the forward pass.

        Parameters
        ----------
        x : Array
            Input tensor. Shape is module-specific; batching is allowed
            (e.g., leading ``(N, ...)`` dimension).
        rng : Array or None, optional
            PRNG key (``KeyArray``). If provided, may be used for stochastic
            behavior; if ``None``, the computation must be deterministic.

        Returns
        -------
        Array
            Output tensor. Shape depends on the module configuration.

        Examples
        --------
        A typical call inside an orchestrator step::

            y = module(x, rng=rng)  # pure function of (x, rng, params, state)

        """
        ...

    @abstractmethod
    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None) -> Self:
        """Compute a local parameter update (same PyTree structure).

        This method implements a **local plasticity rule** that produces a
        PyTree of updates aligned with the module's parameter structure. The
        returned object is typically consumed by an optimizer (e.g., Optax) or
        combined with other updates by the orchestrator.

        Parameters
        ----------
        x : Array
            Inputs seen at the forward call (may be cached by the caller).
        y : Array
            Supervision signal or target associated with the current step.
            Shape must be compatible with the module's output space.
        y_hat : Array
            The module's (or readout's) current prediction.
        gate : Array (optional)
            A multiplicative gate applied to the update. Shape must be
            broadcastable to x shapes.

        Returns
        -------
        Self
            A PyTree with the **same structure** as this module, containing
            per-parameter updates (e.g., ``dW``, ``db``). Implementations may
            return zeros for non-trainable fields.

        Notes
        -----
        The update is *not* required to be a gradient. It can be any local rule
        (e.g., perceptron-style, Hebbian/anti-Hebbian) as long as the shape and
        PyTree layout match the module parameters.

        """
        ...


class Layer(AbstractModule, ABC):
    """A stateful, trainable layer that reduces messages then applies an activation.

    Layers are the primary compute units in Darnax. They typically:
    (1) aggregate incoming messages from upstream modules with :meth:`reduce`,
    and (2) transform the aggregate via :meth:`activation`.

    Notes
    -----
    - Layers are **stateful** by design (e.g., carry hidden activations or
      buffers across steps); the orchestrator is responsible for when/how state
      is read/written. The State object carries the state.
    - The forward pass should be purely functional with respect to parameters
      and external state; do not mutate in-place.

    """

    @property
    def has_state(self) -> bool:
        """Whether the layer carries persistent state.

        Returns
        -------
        bool
            Always ``True`` for layers.

        """
        return True

    @abstractmethod
    def activation(self, x: Array) -> Array:
        """Apply the layer’s activation function.

        Parameters
        ----------
        x : Array
            Pre-activation tensor (e.g., the result of :meth:`reduce`).

        Returns
        -------
        Array
            Post-activation tensor. Typically the same shape as ``x``.

        """
        ...

    @abstractmethod
    def reduce(self, h: PyTree) -> Array:
        """Aggregate incoming messages into a single tensor.

        Parameters
        ----------
        h : PyTree
            Collection of incoming messages from neighbors/upstream modules.
            Implementations define the exact structure; common reducers include
            sum, mean, or structured/sparse contractions.

        Returns
        -------
        Array
            Aggregated input to be passed to :meth:`activation`.

        Examples
        --------
        A sum reducer over message leaves::

            leaves = jax.tree_util.tree_leaves(h)
            x = jnp.sum(jnp.stack(leaves, axis=0), axis=0)
            return x

        """
        ...


class Adapter(AbstractModule, ABC):
    """A stateless mapping between layers.

    Adapters connect layers (e.g., linear projections, reshapes, sparsifying
    maps). They must not carry persistent state and should behave as pure
    functions of inputs and parameters (plus optional RNG).

    Notes
    -----
    - Use adapters to express wiring and shape changes between layers.
    - Because adapters are stateless, orchestration can freely reorder or
      parallelize them as long as data dependencies are respected.

    """

    @property
    def has_state(self) -> bool:
        """Whether the adapter carries persistent state.

        Returns
        -------
        bool
            Always ``False`` for adapters.

        """
        return False
