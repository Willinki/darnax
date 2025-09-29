"""Output layer utilities.

This module defines :class:`OutputLayer`, a lightweight aggregation layer meant
to sit at the boundary of the network. It provides:

- identity **activation** (no nonlinearity),
- identity **forward** pass (conceptually; see Notes),
- a **reduce** method that elementwise-sums all array leaves in a PyTree of
  predictions,
- a **no-op** local update (:meth:`backward`) since there are no trainable
  parameters.

Notes
-----
Despite being conceptually stateless, :class:`OutputLayer` subclasses
:class:`darnax.modules.interfaces.Layer` to reuse the orchestration contract,
including the ``reduce`` interface.

"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Self

import jax

from darnax.modules.interfaces import Layer

if TYPE_CHECKING:
    from jax import Array

    from darnax.utils.typing import PyTree

    KeyArray = Array


class OutputLayer(Layer):
    """Simple output layer that aggregates predictions via summation.

    The layer leaves activations unchanged and defines a ``reduce`` that
    elementwise-sums a PyTree of predictions. Its backward pass is a no-op
    because it has no trainable parameters.

    Notes
    -----
    The current implementation of :meth:`__call__` returns ``zeros_like(x)``.

    """

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Compute the forward pass.

        Parameters
        ----------
        x : Array
            Input tensor; any shape is accepted.
        rng : KeyArray or None, optional
            Ignored; present for signature compatibility.

        Returns
        -------
        Array
            Currently ``zeros_like(x)`` (see Notes in the class docstring).

        """
        return jax.numpy.zeros_like(x)

    def reduce(self, h: PyTree) -> Array:
        """Elementwise-sum all array leaves in a PyTree of predictions.

        Parameters
        ----------
        h : PyTree
            PyTree whose leaves are arrays with identical shapes and dtypes.

        Returns
        -------
        Array
            The elementwise sum across all leaves.

        Raises
        ------
        ValueError
            If ``h`` has no leaves (as per ``tree_reduce`` semantics).

        Notes
        -----
        Uses :func:`jax.tree_util.tree_reduce` with :data:`operator.add`.

        """
        return jax.numpy.asarray(jax.tree_util.tree_reduce(operator.add, h))

    def activation(self, x: Array) -> Array:
        """Identity activation.

        Parameters
        ----------
        x : Array
            Input tensor.

        Returns
        -------
        Array
            ``x`` unchanged.

        """
        return x

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """No-op local update.

        This layer has no trainable parameters, so it returns itself unchanged.

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
            ``self`` (no parameter updates).

        """
        return self
