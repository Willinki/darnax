"""Debug/test modules for Darnax.

This module provides tiny implementations used to exercise the orchestration
and documentation pipeline:

- :class:`DebugLayer`: a stateful layer with a single scalar parameter.
- :class:`DebugAdapter`: a stateless adapter with a single scalar parameter.

They are intentionally simple and **not** meant for learning quality; they
exist to validate wiring, typing, and update flows.
"""

import operator
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from darnax.modules.interfaces import Adapter, Layer
from darnax.utils.typing import PyTree


class DebugLayer(Layer):
    """A minimal trainable layer for testing and debugging.

    Multiplies inputs elementwise by a learnable scalar ``w``. The activation
    (:meth:`activation`) is the sign function, but the default forward pass
    (:meth:`__call__`) deliberately returns ``w * x`` to keep behavior simple.

    Attributes
    ----------
    w : Array
        Learnable weight, shape ``(1,)``, dtype ``float32``.
    a : bool
        Static flag (non-PyTree) to mark the object as a layer. Kept for
        debugging/demo purposes only.

    Notes
    -----
    - This class is **stateful** (as all :class:`~darnax.modules.interfaces.Layer`).
    - The local update (:meth:`backward`) is a no-op that returns ``self``; it
      exists to exercise the training loop, not to learn.

    """

    w: Array
    a: bool = eqx.field(static=True)

    def __init__(self) -> None:
        """Initialize parameters.

        Notes
        -----
        Sets ``w = 1.0`` (``float32``) and ``a = True``.

        """
        self.w = jnp.ones((1,), dtype=jnp.float32)
        self.a = True

    def __call__(self, x: Array, rng: DTypeLike | None = None) -> Array:
        """Compute the forward pass.

        Parameters
        ----------
        x : Array
            Input tensor. Must be broadcastable with ``(1,)``.
        rng : DTypeLike or None, optional
            Ignored; present for signature compatibility.

        Returns
        -------
        Array
            Elementwise product ``w * x``.

        """
        return self.w * x

    def activation(self, x: Array) -> Array:
        """Apply the layer’s activation function.

        Parameters
        ----------
        x : Array
            Pre-activation tensor.

        Returns
        -------
        Array
            ``sign(x)``.

        """
        return jnp.sign(x)

    def reduce(self, h: PyTree) -> Array:
        """Aggregate incoming messages into a single tensor.

        Parameters
        ----------
        h : PyTree
            PyTree of arrays to be summed.

        Returns
        -------
        Array
            Sum over all leaves in ``h`` via an associative reduction.

        Notes
        -----
        Uses :func:`jax.tree.reduce_associative` with ``operator.add``.

        """
        return jnp.asarray(jax.tree.reduce_associative(operator.add, h))

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """Return a no-op parameter update.

        Parameters
        ----------
        x : Array
            Forward input (unused).
        y : Array
            Target/supervision (unused).
        y_hat : Array
            Prediction (unused).

        Returns
        -------
        Self
            ``self`` — identity update (no change).

        """
        return self


class DebugAdapter(Adapter):
    """A minimal stateless mapping for testing and debugging.

    Multiplies inputs elementwise by a scalar ``w``. Carries no persistent
    state and returns a no-op update in :meth:`backward`.

    Attributes
    ----------
    w : Array
        Weight, shape ``(1,)``, dtype ``float32``.
    a : bool
        Static flag (non-PyTree) to mark the object as an adapter.

    """

    w: Array
    a: bool = eqx.field(static=True)

    def __init__(self) -> None:
        """Initialize parameters.

        Notes
        -----
        Sets ``w = 1.0`` (``float32``) and ``a = False``.

        """
        self.w = jnp.ones((1,), dtype=jnp.float32)
        self.a = False

    def __call__(self, x: Array, rng: DTypeLike | None = None) -> Array:
        """Compute the forward pass.

        Parameters
        ----------
        x : Array
            Input tensor. Must be broadcastable with ``(1,)``.
        rng : DTypeLike or None, optional
            Ignored; present for signature compatibility.

        Returns
        -------
        Array
            Elementwise product ``w * x``.

        """
        return self.w * x

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """Return a no-op parameter update.

        Parameters
        ----------
        x : Array
            Forward input (unused).
        y : Array
            Target/supervision (unused).
        y_hat : Array
            Prediction (unused).

        Returns
        -------
        Self
            ``self`` — identity update (no change).

        """
        return self
