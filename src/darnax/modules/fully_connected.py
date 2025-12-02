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

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Return a module-shaped local update where only ``ΔW`` is set.

        Parameters
        ----------
        x : Array
            Forward input(s), shape ``(..., in_features)``.
        y : Array
            Supervision signal/targets, broadcast-compatible with ``y_hat``.
        y_hat : Array
            Current prediction/logits, broadcast-compatible with ``y``.
        gate : Array
            Multiplicative gate applied to the update, default is ``1.0``.

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
        if gate is None:
            gate = jnp.array(1.0)
        dW = perceptron_rule_backward(x, y, y_hat, self.threshold, gate)
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

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Return zero update for all parameters.

        Parameters
        ----------
        x : Array
            Forward input (unused).
        y : Array
            Target/supervision (unused).
        y_hat : Array
            Prediction/logits (unused).
        gate : Array
            Multiplicative gate (unused).

        Returns
        -------
        Self
            PyTree of zeros with the same structure as ``self``.

        """
        zero_update: Self = jax.tree.map(jnp.zeros_like, self)
        return zero_update


class SparseFullyConnected(FullyConnected):
    """Fully connected adapter with a fixed binary sparsity mask.

    This variant samples a Bernoulli mask at initialization time and applies it
    multiplicatively to both the weights and their updates. Forward and local
    learning behave like :class:`FullyConnected`, but gradients/updates are
    constrained to the active connections.

    Attributes
    ----------
    W : Array
        Weight matrix with shape ``(in_features, out_features)``, masked by
        ``_mask``.
    strength : Array
        Per-output scale, shape ``(out_features,)``; used in the same way as in
        :class:`FullyConnected` for initialization.
    threshold : Array
        Per-output margin used by the local update rule, shape
        ``(out_features,)``.
    _mask : Array
        Binary mask with shape ``(in_features, out_features)`` indicating which
        connections are active (ones) or pruned (zeros).

    Notes
    -----
    The mask is sampled once at initialization and kept fixed. Updates produced
    by :meth:`backward` are re-masked so that pruned connections remain zero.

    """

    _mask: Array

    def __init__(
        self,
        in_features: int,
        out_features: int,
        strength: float | ArrayLike,
        threshold: float | ArrayLike,
        sparsity: float,
        key: Array,
        dtype: DTypeLike = jnp.float32,
    ):
        """Initialize sparse mask and masked weight matrix.

        Parameters
        ----------
        in_features : int
            Input dimensionality.
        out_features : int
            Output dimensionality.
        strength : float or ArrayLike
            Scalar (broadcast to ``(out_features,)``) or a vector of length
            ``out_features`` providing the per-output scaling used in weight
            initialization.
        threshold : float or ArrayLike
            Scalar or vector of length ``out_features`` with the per-output
            margins used by the local update rule.
        sparsity : float
            Fraction of entries in ``W`` that are set to zero on average, in the
            interval ``[0.0, 1.0)``. The mask keeps a fraction ``1 - sparsity``
            of connections.
        key : Array
            JAX PRNG key used to sample both the weight matrix and the mask.
        dtype : DTypeLike, optional
            Dtype for parameters (default: ``jnp.float32``).

        Notes
        -----
        The weight matrix is initialized with Gaussian entries scaled by
        ``strength / sqrt(in_features * (1 - sparsity))`` and then multiplied
        by the sampled Bernoulli mask.

        """
        self.strength = self._set_shape(strength, out_features, dtype)
        self.threshold = self._set_shape(threshold, out_features, dtype)
        key_w, key_mask = jax.random.split(key)
        mask = jax.random.bernoulli(
            key_mask,
            p=1.0 - sparsity,
            shape=(in_features, out_features),
        )
        W = (
            jax.random.normal(key_w, shape=(in_features, out_features), dtype=dtype)
            * self.strength
            / jnp.sqrt(in_features * (1 - sparsity))
        )
        self._mask = mask
        self.W = W * mask

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Return a masked local update where only ``ΔW`` is nonzero.

        Parameters
        ----------
        x : Array
            Forward input(s), shape ``(..., in_features)``.
        y : Array
            Supervision signal/targets, broadcast-compatible with ``y_hat``.
        y_hat : Array
            Current prediction/logits, broadcast-compatible with ``y``.
        gate : Array or None, optional
            Optional multiplicative gate applied elementwise to the update.
            If ``None``, the gate is passed through unchanged to
            :func:`perceptron_rule_backward`.

        Returns
        -------
        Self
            A PyTree with the same structure as ``self`` where:
            - ``W`` holds the masked update ``ΔW * _mask``,
            - ``strength``, ``threshold``, and ``_mask`` leaves are zeros.

        Notes
        -----
        The unmasked update is first computed via
        :func:`darnax.utils.perceptron_rule.perceptron_rule_backward` and then
        multiplied by the fixed binary mask so that pruned connections are
        never updated.

        """
        dW = perceptron_rule_backward(x, y, y_hat, self.threshold, gate)
        dW = dW * self._mask
        zero_update = jax.tree.map(jnp.zeros_like, self)
        new_self: Self = eqx.tree_at(lambda m: m.W, zero_update, dW)
        return new_self


class FrozenRescaledFullyConnected(FullyConnected):
    """Fully connected adapter with **frozen** parameters.

    Same forward behavior as :class:`FullyConnected`, but :meth:`backward`
    returns **zeros** for all leaves.
    Usually used to propagate information back from the label to internal layers.
    Before projecting, it applies the following rescaling ``+1 -> sqrt(C-1); -1 -> 1/sqrt(C-1)``,
    where C is the input dimension (usually number of classes).
    """

    def __call__(self, y: Array, rng: KeyArray | None = None) -> Array:
        """Compute ``y = (x @ W) * strength`` (broadcast on last dim).

        Parameters
        ----------
        y : Array
            Input tensor with trailing dimension ``in_features``. Leading batch
            dimensions (e.g., ``(N, ...)``) are supported via standard matmul
            broadcasting. We expect that ``in_features`` equals the number of classes,
            and the entries are one-hot encoded (-1/+1).
        rng : KeyArray or None, optional
            Ignored; present for signature compatibility.

        Returns
        -------
        Array
            Output tensor with trailing dimension ``out_features``.

        """
        C = self.W.shape[0]
        Cr_m1 = (C - 1) ** 0.5
        a = 1 / 2 * (Cr_m1 / 2 + 1 / Cr_m1)
        b = 1 / 2 * (Cr_m1 / 2 - 1 / Cr_m1)
        return jnp.asarray((y * a + b) @ self.W)

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Return zero update for all parameters.

        Parameters
        ----------
        x : Array
            Forward input (unused).
        y : Array
            Target/supervision (unused).
        y_hat : Array
            Prediction/logits (unused).
        gate : Array
            Multiplicative gate (unused).

        Returns
        -------
        Self
            PyTree of zeros with the same structure as ``self``.

        """
        zero_update: Self = jax.tree.map(jnp.zeros_like, self)
        return zero_update


class CEFullyConnected(FullyConnected):
    """Fully connected adapter with cross-entropy-style local updates.

    This variant shares the same forward map as :class:`FullyConnected` but
    overrides :meth:`backward` to implement the local gradient of a softmax +
    cross-entropy loss with respect to ``W``. Targets are assumed to be in
    ``{-1, +1}`` and are internally mapped to one-hot vectors in ``{0, 1}``.

    Notes
    -----
    - Only ``W`` is updated; ``strength`` and ``threshold`` are left at zero in
      the returned update PyTree.
    - The update is normalized by both the batch size and the input dimension,
      to match the scale convention used by the perceptron-style rule.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        strength: float | ArrayLike,
        threshold: float | ArrayLike,
        key: Array,
        dtype: DTypeLike = jnp.float32,
    ):
        """Initialize weights and hyperparameters for CE-based updates.

        Parameters
        ----------
        in_features : int
            Input dimensionality.
        out_features : int
            Output dimensionality (number of classes).
        strength : float or ArrayLike
            Scalar or vector of length ``out_features`` providing the per-output
            scaling used in weight initialization.
        threshold : float or ArrayLike
            Unused by the CE update but kept for API compatibility with
            :class:`FullyConnected`.
        key : Array
            JAX PRNG key used to initialize ``W``.
        dtype : DTypeLike, optional
            Dtype for parameters (default: ``jnp.float32``).

        Notes
        -----
        Initialization is delegated to :class:`FullyConnected` via ``super()``.

        """
        super().__init__(in_features, out_features, strength, threshold, key, dtype)

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        r"""Return a CE/softmax local gradient update for ``W``.

        Parameters
        ----------
        x : Array
            Forward inputs, shape ``(B, in_features)`` where ``B`` is the batch
            size.
        y : Array
            Targets in ``{-1, +1}``, shape ``(B, out_features)``. Values are
            internally mapped to one-hot class indicators via
            ``(y + 1) / 2``.
        y_hat : Array
            Logits or pre-softmax activations, shape ``(B, out_features)``.
        gate : Any or None, optional
            Currently ignored; present for signature compatibility with the
            :class:`Adapter` interface.

        Returns
        -------
        Self
            A PyTree with the same structure as ``self`` where:
            - ``W`` holds the CE-based update ``ΔW``,
            - ``strength`` and ``threshold`` leaves are zeros.

        Notes
        -----
        The update implements

        .. math::

            p &= \\operatorname{softmax}(y_\\text{hat}) \\\\
            \\Delta W &\\propto x^\\top (p - y_\\text{onehot})

        scaled by the batch size and by ``1 / sqrt(in_features)`` so that its
        magnitude is comparable to the perceptron-style rule used in
        :class:`FullyConnected`.

        """
        # local gradient of cross-entropy loss with softmax
        # assuming y in {-1, +1}
        B = y_hat.shape[0]
        H = self.W.shape[0]
        probs = jax.nn.softmax(y_hat, axis=-1)  # B, C
        dL_dz = probs - (y + 1) / 2  # B, C
        dL_dW = x.T @ dL_dz / B  # H, C
        dW = dL_dW / (H**0.5)  # same convention as perceptron rule
        zero_update = jax.tree.map(jnp.zeros_like, self)
        new_self: Self = eqx.tree_at(lambda m: m.W, zero_update, dW)
        return new_self
