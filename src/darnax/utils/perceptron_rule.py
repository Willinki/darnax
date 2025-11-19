"""Perceptron (OVA) backward update in JAX.

This module implements a one-vs-all perceptron-style local update that returns
a weight increment ``ΔW`` to be **added** to a dense weight matrix.

- Supports batched inputs.
- Labels are in ``{-1, +1}`` per class (one-vs-all coding).
- ``y_hat`` are raw scores (pre-activation).
- No learning rate is applied here.

Shapes
------
x       : (d,) or (n, d)
y       : (n, K) in {-1, +1}
y_hat   : (n, K) raw scores
margin  : broadcastable to (n, K) — e.g. scalar, (K,), or (n, K)
returns : (d, K)  (ΔW to add to weights)

"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def perceptron_rule_backward(
    x: jax.Array,
    y: jax.Array,
    y_hat: jax.Array,
    margin: jax.Array,
    gate: jax.Array | None = None,
) -> jax.Array:
    """Multiclass (OVA) perceptron update (no learning rate).

    Applies an update when the margin condition is violated,
    i.e. ``y * y_hat <= margin`` (ties count as mistakes). The result is a
    weight increment ``ΔW`` aligned with a column-per-class layout.

    Parameters
    ----------
    x : jax.Array
        Input vector(s), shape ``(d,)`` or ``(n, d)``.
    y : jax.Array
        One-vs-all targets in ``{-1, +1}``, shape ``(n, K)``.
    y_hat : jax.Array
        Raw scores (pre-activation), shape ``(n, K)``.
    margin : jax.Array
        Margin threshold, **broadcastable** to ``(n, K)``; e.g. a scalar,
        a per-class vector ``(K,)``, or an array ``(n, K)``.
    gate: jax.Array
        Multiplicative gate applied to the update, default is ``1.0``.
        Should have shape broadcastable to x shape.

    Returns
    -------
    jax.Array
        Weight update ``ΔW`` of shape ``(d, K)`` to **add** to the weights.

    Notes
    -----
    - Batch-size normalization: the update is divided by ``n**0.5`` so that its
      magnitude is invariant to the batch size.
    - Fan-in/width normalization (``1/sqrt(d)``) is applied here.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> x = jnp.array([[1., 0.], [0., 1.]])      # (n=2, d=2)
    >>> y = jnp.array([[+1, -1], [-1, +1]])      # (2, K=2)
    >>> y_hat = jnp.array([[0.2, -0.3], [0.1, -0.4]])
    >>> margin = 0.0
    >>> dW = perceptron_rule_backward(x, y, y_hat, margin)
    >>> dW.shape
    (2, 2)

    """
    if gate is None:
        gate = jnp.array(1.0)
    x = jnp.atleast_2d(x)  # (n, d)
    y = jnp.atleast_2d(y)  # (n, K)
    y_hat = jnp.atleast_2d(y_hat)  # (n, K)

    n, d = x.shape
    if y.shape != y_hat.shape or y.shape[0] != n:
        raise ValueError("y and y_hat must have the same (n, K) shape.")
    m = y * y_hat  # (n, K)
    mistake = (m <= margin).astype(x.dtype)  # (n, K)
    update: jax.Array = (x.T * gate.T @ (mistake * y)) / (n**0.5 * d**0.5)  # (d, K)
    return -update
