"""Binary (±1) recurrent layer with dense couplings.

This module defines :class:`RecurrentDiscrete`, a fully connected recurrent
layer whose states live in ``{-1, +1}``. The coupling matrix ``J`` has a
controllable diagonal ``J_D`` (self-couplings), and learning uses a local,
perceptron-style rule with per-unit margins (``threshold``).

Design
------
- **State space:** discrete, ``s_i ∈ {-1, +1}``.
- **Couplings:** dense matrix ``J ∈ ℝ^{d×d}``, diagonal forced to ``J_D``.
- **Forward:** pre-activation ``h = x @ J`` (no in-place mutation).
- **Activation:** hard sign with ties to ``+1``.
- **Learning:** :func:`darnax.utils.perceptron_rule.perceptron_rule_backward`
  produces ``ΔJ``; the **diagonal update is masked to zero** so self-couplings
  remain fixed at ``J_D``.

Notes
-----
This class is an Equinox ``Module`` (a PyTree). Parameters are leaves and can be
updated via Optax or custom update code. The orchestrator controls when to call
``activation`` vs forwarding pre-activations.

See Also
--------
darnax.modules.interfaces.Layer
darnax.utils.perceptron_rule.perceptron_rule_backward

"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import tree_reduce

from darnax.modules.interfaces import Layer
from darnax.utils.perceptron_rule import perceptron_rule_backward

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import ArrayLike, DTypeLike

    from darnax.utils.typing import PyTree

    KeyArray = jax.Array


class RecurrentDiscrete(Layer):
    """Binary (±1) recurrent layer with dense couplings.

    The layer keeps a dense coupling matrix ``J`` (with fixed diagonal
    ``J_D``), a per-unit margin ``threshold`` for local updates, and an
    internal diagonal mask used to zero out self-updates during learning.

    Attributes
    ----------
    J : Array
        Coupling matrix with shape ``(features, features)``.
    J_D : Array
        Diagonal self-couplings, shape ``(features,)``. Mirrors
        ``jnp.diag(J)`` and is kept fixed by masking during updates.
    threshold : Array
        Per-unit margin used by the local perceptron-style rule,
        shape ``(features,)``.
    strength: float, default=1.0
        Scalar that multiplies all couplings at initialization to increase layer
        influence in the dynamics. Similar to strengths in fully connected
        and ferromagnetic adapters.
    _mask : Array
        Binary matrix (``1 - I``) that zeroes the diagonal of ``ΔJ`` before
        applying updates. Same shape and dtype as ``J``.

    """

    J: Array
    J_D: Array
    threshold: Array
    strength: Array
    _mask: Array

    def __init__(
        self,
        features: int,
        j_d: ArrayLike,
        threshold: ArrayLike,
        key: KeyArray,
        strength: float = 1.0,
        dtype: DTypeLike = jnp.float32,
    ):
        """Construct the layer parameters.

        Initializes a dense coupling matrix ``J`` with i.i.d. Gaussian entries
        scaled by ``1/sqrt(features)`` and sets its diagonal to ``j_d``.
        Stores per-unit margins in ``threshold`` and a diagonal masking matrix
        to keep self-couplings fixed during learning.

        Parameters
        ----------
        features : int
            Number of units (dimension ``d``). Shapes are derived from this.
        j_d : ArrayLike
            Self-couplings (diagonal of ``J``). Either a scalar (broadcast to
            ``(features,)``) or a vector of length ``features``.
        threshold : ArrayLike
            Per-unit margins for the local update rule. Scalar or vector of
            length ``features``.
        key : KeyArray
            JAX PRNG key used to initialize the off-diagonal entries of ``J``.
        strength: float, optional
            Scalar that multiplies all couplings at initialization to increase layer
            influence in the dynamics. Similar to strengths in fully connected
            and ferromagnetic adapters.
        dtype : DTypeLike, optional
            Parameter dtype, by default ``jnp.float32``.

        Raises
        ------
        ValueError
            If ``j_d`` or ``threshold`` is not scalar or a 1D vector with
            length ``features``.

        """
        j_d_vec = self._set_shape(j_d, features, dtype)
        thresh_vec = self._set_shape(threshold, features, dtype)
        strength_vec = jnp.asarray(strength, dtype=dtype)

        J = (
            jax.random.normal(key, shape=(features, features), dtype=dtype)
            / jnp.sqrt(features)
            * strength_vec
        )
        diag = jnp.diag_indices(features)
        J = J.at[diag].set(j_d_vec)

        self.J = J
        self.J_D = j_d_vec
        self.threshold = thresh_vec
        self.strength = strength_vec
        self._mask = 1 - jnp.eye(features, dtype=dtype)

    def activation(self, x: Array) -> Array:
        """Hard-sign activation mapping ties to ``+1``.

        Parameters
        ----------
        x : Array
            Pre-activation tensor.

        Returns
        -------
        Array
            ``(+1)`` where ``x >= 0`` and ``(-1)`` otherwise, cast to
            ``x.dtype``.

        Notes
        -----
        This function is separate from :meth:`__call__` so orchestrators can
        decide when to discretize (e.g., training vs inference dynamics).

        """
        return jnp.where(x >= 0, 1, -1).astype(x.dtype)

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        r"""Compute pre-activations.

        Performs a dense update:

        .. math::
            h = x \\cdot J

        Parameters
        ----------
        x : Array
            Input state(s). Shape ``(features,)`` or ``(batch, features)``.
        rng : KeyArray or None, optional
            Ignored; present for signature compatibility.

        Returns
        -------
        Array
            Pre-activation tensor with shape ``(features,)`` or
            ``(batch, features)`` matching ``x``.

        """
        return x @ self.J

    def reduce(self, h: PyTree) -> Array:
        """Aggregate incoming messages by summation.

        Parameters
        ----------
        h : PyTree
            PyTree of arrays (e.g., messages from neighbors) to be summed.

        Returns
        -------
        Array
            Elementwise sum over all leaves in ``h``.

        Notes
        -----
        Uses :func:`jax.tree_util.tree_reduce` with :data:`operator.add`.

        """
        return jnp.asarray(tree_reduce(operator.add, h))

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Compute a module-shaped local update.

        Produces a PyTree of updates where only ``J`` receives a nonzero
        ``ΔJ``; all other fields are zero. The diagonal of ``ΔJ`` is masked
        to zero so self-couplings stay fixed at ``J_D``.

        Parameters
        ----------
        x : Array
            Inputs used to produce the current prediction. Shape
            ``(features,)`` or ``(batch, features)``.
        y : Array
            Supervision signal/targets, broadcast-compatible with ``y_hat``.
        y_hat : Array
            Current prediction/logits, broadcast-compatible with ``y``.
        gate : Array
            A multiplicative gate applied to the update. Shape must be
            broadcastable to x shapes.

        Returns
        -------
        Self
            A PyTree with the same structure as ``self`` where:
            - ``J`` contains ``ΔJ`` (diagonal zeroed),
            - all other leaves are zeros.

        Notes
        -----
        Calls :func:`darnax.utils.perceptron_rule.perceptron_rule_backward`
        with the stored per-unit ``threshold``. The rule is local and need not
        be a true gradient.

        Examples
        --------
        >>> upd = layer.backward(x, y, y_hat)
        >>> new_params = eqx.tree_at(lambda m: m.J, layer, layer.J + lr * upd.J)

        """
        if gate is None:
            gate = jnp.array(1.0)
        dJ = perceptron_rule_backward(x, y, y_hat, self.threshold, gate)
        dJ = dJ * self._mask
        zero_update = jax.tree.map(jnp.zeros_like, self)
        new_self: Self = eqx.tree_at(lambda m: m.J, zero_update, dJ)
        return new_self

    @staticmethod
    def _set_shape(x: ArrayLike, dim: int, dtype: DTypeLike) -> Array:
        """Normalize a scalar or vector to shape ``(dim,)`` and dtype.

        Parameters
        ----------
        x : ArrayLike
            Scalar or 1D array.
        dim : int
            Expected length for 1D input or broadcasted scalar.
        dtype : DTypeLike
            Target dtype.

        Returns
        -------
        Array
            A vector of shape ``(dim,)`` with dtype ``dtype``.

        Raises
        ------
        ValueError
            If ``x`` is neither scalar nor a 1D array of length ``dim``.

        """
        x = jnp.array(x, dtype)
        if x.ndim == 0:
            return jnp.broadcast_to(x, (dim,))
        if x.ndim == 1:
            if x.shape[0] != dim:
                raise ValueError(f"length {x.shape[0]} != features {dim}")
            return x
        raise ValueError("expected scalar or 1D vector")
