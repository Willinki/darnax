"""Fully connected adapters.

This module provides two Equinox-based adapters:

- :class:`FullyConnected`: trainable affine map with per-output scaling and
  a local perceptron-style update (only ``W`` is updated).
- :class:`FrozenFullyConnected`: same forward as ``FullyConnected`` but returns
  a zero update (useful for inference or ablation).

Both classes are **stateless** in the runtime sense (no persistent state across
steps) but **parameterized** (PyTrees with trainable weights).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import equinox as eqx
import jax
import jax.numpy as jnp

from darnax.modules.interfaces import Adapter
from darnax.utils.perceptron_rule import perceptron_rule_backward

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import ArrayLike, DTypeLike

    KeyArray = Array


class FullyConnected(Adapter):
    """Fully connected trainable adapter ``y = (x @ W) * strength``.

    A dense linear projection followed by an elementwise per-output scaling.
    Learning uses a **local perceptron-style rule** parameterized by a
    per-output ``threshold``; only ``W`` receives updates, while
    ``strength`` and ``threshold`` act as (learnable-if-you-want) hyperparameters
    that are **not** updated by :meth:`backward`.

    Attributes
    ----------
    W : Array
        Weight matrix with shape ``(in_features, out_features)``.
    strength : Array
        Per-output scale, shape ``(out_features,)``; broadcast across the last
        dimension of the forward output.
    threshold : Array
        Per-output margin used by the local update rule, shape ``(out_features,)``.

    Notes
    -----
    - Adapters are *stateless* per the Darnax interface, but they may carry
      trainable parameters. This class advertises trainability through ``W``.
    - The local rule is supplied by
      :func:`darnax.utils.perceptron_rule.perceptron_rule_backward` and is not
      required to be a gradient.

    """

    W: Array
    strength: Array
    threshold: Array

    def __init__(
        self,
        in_features: int,
        out_features: int,
        strength: float | ArrayLike,
        threshold: float | ArrayLike,
        key: Array,
        dtype: DTypeLike = jnp.float32,
    ):
        """Initialize weights and per-output scale/threshold.

        Parameters
        ----------
        in_features : int
            Input dimensionality.
        out_features : int
            Output dimensionality.
        strength : float or ArrayLike
            Scalar (broadcast to ``(out_features,)``) or a vector of
            length ``out_features`` providing the per-output scaling.
        threshold : float or ArrayLike
            Scalar or vector of length ``out_features`` with the per-output
            margins used by the local update rule.
        key : Array
            JAX PRNG key to initialize ``W`` with Gaussian entries scaled by
            ``1/sqrt(in_features)``.
        dtype : DTypeLike, optional
            Dtype for parameters (default: ``jnp.float32``).

        Raises
        ------
        ValueError
            If ``strength`` or ``threshold`` is neither a scalar nor a 1D array
            of the expected length.

        """
        self.strength = self._set_shape(strength, out_features, dtype)
        self.threshold = self._set_shape(threshold, out_features, dtype)
        self.W = (
            jax.random.normal(key, (in_features, out_features), dtype=dtype)
            * self.strength
            / jnp.sqrt(in_features)
        )

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Compute ``y = (x @ W) * strength`` (broadcast on last dim).

        Parameters
        ----------
        x : Array
            Input tensor with trailing dimension ``in_features``. Leading batch
            dimensions (e.g., ``(N, ...)``) are supported via standard matmul
            broadcasting.
        rng : KeyArray or None, optional
            Ignored; present for signature compatibility.

        Returns
        -------
        Array
            Output tensor with trailing dimension ``out_features``.

        """
        return x @ self.W

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """Return a module-shaped local update where only ``ΔW`` is set.

        Parameters
        ----------
        x : Array
            Forward input(s), shape ``(..., in_features)``.
        y : Array
            Supervision signal/targets, broadcast-compatible with ``y_hat``.
        y_hat : Array
            Current prediction/logits, broadcast-compatible with ``y``.

        Returns
        -------
        Self
            A PyTree with the same structure as ``self`` where:
            - ``W`` holds the update ``ΔW`` from the local rule,
            - ``strength`` and ``threshold`` leaves are zeros.

        Notes
        -----
        Calls :func:`darnax.utils.perceptron_rule.perceptron_rule_backward`
        with the stored per-output ``threshold``.

        """
        dW = perceptron_rule_backward(x, y, y_hat, self.threshold)
        zero_update = jax.tree.map(jnp.zeros_like, self)
        new_self: Self = eqx.tree_at(lambda m: m.W, zero_update, dW)
        return new_self

    @staticmethod
    def _set_shape(x: ArrayLike, dim: int, dtype: DTypeLike) -> Array:
        """Normalize scalar or 1D input to shape ``(dim,)`` and dtype.

        Parameters
        ----------
        x : ArrayLike
            Scalar or 1D array.
        dim : int
            Expected length for a 1D input or broadcasted scalar.
        dtype : DTypeLike
            Target dtype.

        Returns
        -------
        Array
            Vector of shape ``(dim,)`` and dtype ``dtype``.

        Raises
        ------
        ValueError
            If ``x`` is neither scalar nor a 1D array with length ``dim``.

        """
        x = jnp.array(x, dtype=dtype)
        if x.ndim == 0:
            return jnp.broadcast_to(x, (dim,))
        if x.ndim == 1:
            if x.shape[0] != dim:
                raise ValueError(f"length {x.shape[0]} != expected {dim}")
            return x
        raise ValueError("expected scalar or 1D vector")


class FrozenFullyConnected(FullyConnected):
    """Fully connected adapter with **frozen** parameters.

    Same forward behavior as :class:`FullyConnected`, but :meth:`backward`
    returns **zeros** for all leaves. Useful for inference-only deployments
    or to ablate learning of a particular edge type in a graph.
    """

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        """Return zero update for all parameters.

        Parameters
        ----------
        x : Array
            Forward input (unused).
        y : Array
            Target/supervision (unused).
        y_hat : Array
            Prediction/logits (unused).

        Returns
        -------
        Self
            PyTree of zeros with the same structure as ``self``.

        """
        zero_update: Self = jax.tree.map(jnp.zeros_like, self)
        return zero_update
