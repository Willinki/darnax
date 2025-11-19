"""Adapters: fixed linear couplings.

This module defines :class:`Ferromagnetic`, a stateless adapter that scales
signals elementwise by a fixed coupling ``strength`` (scalar or per-feature).

Adapters in Darnax are Equinox ``Module``s and thus PyTrees; they should not
hold persistent state and typically return zero updates in :meth:`backward`.
"""

from typing import Self

import jax.numpy as jnp
from jax import Array
from jax.tree_util import tree_map
from jax.typing import ArrayLike, DTypeLike

from darnax.modules.interfaces import Adapter

KeyArray = Array


class Ferromagnetic(Adapter):
    """Elementwise scaling adapter with fixed coupling strength.

    Multiplies inputs by a constant coupling vector ``strength``. This module
    is **stateless** and **non-trainable**: its :meth:`backward` returns a
    zero-shaped update PyTree.

    Attributes
    ----------
    strength : Array
        Coupling strengths, shape ``(features,)``; may originate from a scalar
        broadcast to that shape at construction time.

    """

    strength: Array

    def __init__(self, features: int, strength: ArrayLike, dtype: DTypeLike = jnp.float32):
        """Construct the adapter with scalar or per-feature strength.

        Parameters
        ----------
        features : int
            Number of features; determines the length of ``strength``.
        strength : ArrayLike
            Either a scalar (broadcast to ``(features,)``) or a 1D array of
            length ``features``.
        dtype : DTypeLike, optional
            Dtype for the internal ``strength`` array. Default is ``float32``.

        Raises
        ------
        ValueError
            If ``strength`` is neither a scalar nor a 1D array of length
            ``features``.

        """
        self.strength = self._set_shape(strength, features, dtype)

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Scale inputs elementwise by the coupling strength.

        Parameters
        ----------
        x : Array
            Input tensor whose trailing dimension is broadcast-compatible with
            ``(features,)``.
        rng : KeyArray or None, optional
            Ignored; present for signature compatibility.

        Returns
        -------
        Array
            ``x * strength`` with standard NumPy/JAX broadcasting.

        """
        return x * self.strength

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Return a zero update to indicate non-trainability.

        Parameters
        ----------
        x : Array
            Forward input (unused).
        y : Array
            Target/supervision (unused).
        y_hat : Array
            Prediction (unused).
        gate : Array
            Multiplicative gate (unused).

        Returns
        -------
        Self
            A PyTree matching ``self`` where all leaves are zeros.

        """
        if gate is None:
            gate = jnp.array(1.0)
        new_self: Self = tree_map(jnp.zeros_like, self)
        return new_self

    @staticmethod
    def _set_shape(x: ArrayLike, features: int, dtype: DTypeLike) -> Array:
        """Normalize ``strength`` to a vector of shape ``(features,)``.

        Parameters
        ----------
        x : ArrayLike
            Scalar or 1D array.
        features : int
            Expected length for 1D input or broadcasted scalar.
        dtype : DTypeLike
            Target dtype for the resulting array.

        Returns
        -------
        Array
            Strength vector of shape ``(features,)`` and dtype ``dtype``.

        Raises
        ------
        ValueError
            If ``x`` is neither scalar nor a 1D array of length ``features``.

        """
        x = jnp.array(x, dtype)
        if x.ndim == 0:
            return jnp.broadcast_to(x, (features,))
        if x.ndim == 1:
            if x.shape[0] != features:
                raise ValueError(f"strength length {x.shape[0]} != features {features}")
            return x
        raise ValueError("strength must be a scalar or 1D vector")
