"""Continouos Perceptron (OVA) backward update in JAX.

This module implements a one-vs-all perceptron-style local update that returns
a weight increment ``ΔW`` to be **added** to a dense weight matrix.

- Supports batched inputs.
- Labels are continouos in ``[-1, +1]`` per class (one-vs-all coding).
- ``y_hat`` are raw scores (pre-activation).
- No learning rate is applied here.

Shapes
------
x       : (d,) or (n, d)
y       : (n, K) in [-1, +1]
y_hat   : (n, K) raw scores
margin  : broadcastable to (n, K) — e.g. scalar, (K,), or (n, K)
returns : (d, K)  (ΔW to add to weights)

"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def tanh_perceptron_rule_backward(
    x: jax.Array,
    y: jax.Array,
    y_hat: jax.Array,
    tolerance: jax.Array,
) -> jax.Array:
    """Multiclass (OVA) perceptron update (no learning rate).

    Applies an update when the tolerance condition is violated,
    i.e. ``y - y_hat <= tolerance`` (ties count as mistakes). The result is a
    weight increment ``ΔW`` aligned with a column-per-class layout.

    Parameters
    ----------
    x : jax.Array
        Input vector(s), shape ``(d,)`` or ``(n, d)``.
    y : jax.Array
        One-vs-all targets in ``[-1, +1]``, shape ``(n, K)``.
    y_hat : jax.Array
        Pre-activations xW, shape ``(n, K)``.
    tolerance : jax.Array
        Tolerance. Weights are update only if error > tolerance,
        **broadcastable** to ``(n, K)``; e.g. a scalar,
        a per-class vector ``(K,)``, or an array ``(n, K)``.

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
    # Ensure batch dimension
    x = jnp.atleast_2d(x)  # (n, d)
    n, d = x.shape

    # Model output (post-activation);
    o = jnp.tanh(y_hat)  # (n, K)
    # update only when |error| >= tolerance)
    err = y - o  # (n, K)
    tol = jnp.broadcast_to(tolerance, err.shape)
    mask = (jnp.abs(err) >= tol).astype(x.dtype)
    # delta rule with tanh derivative: ΔW ∝ x^T @ [(y - o) * (1 - o^2)]
    local = err * (1.0 - o**2) * mask  # (n, K)
    dW = x.T @ local  # (d, K)
    # Normalizations: batch-size (√n) and fan-in/width (√d)
    dW = dW / (jnp.sqrt(n) * jnp.sqrt(d))
    return -dW


def tanh_truncated_perceptron_rule_backward(
    x: jax.Array,
    y: jax.Array,
    y_hat: jax.Array,
    margin: jax.Array,
    tolerance: jax.Array,
) -> jax.Array:
    """Multiclass (OVA) perceptron update (no learning rate).

    It constitutes a variant of the rule for discrete units.
    Here we have continuous units, with tanh activation, that
    are treated as discrete if 1-|s_i| < tolerance, and ignored
    otherwise.
    Applies an update when the margin condition is violated,
    i.e. ``y * y_hat <= margin`` (ties count as mistakes). The result is a
    weight increment ``ΔW`` aligned with a column-per-class layout.


    Parameters
    ----------
    x : jax.Array
        Input vector(s), shape ``(d,)`` or ``(n, d)``.
    y : jax.Array
        One-vs-all targets in ``[-1, +1]``, shape ``(n, K)``.
    y_hat : jax.Array
        Raw scores (pre-activation), shape ``(n, K)``.
    margin : jax.Array
        Margin threshold, **broadcastable** to ``(n, K)``; e.g. a scalar,
        a per-class vector ``(K,)``, or an array ``(n, K)``.
    tolerance : jax.Array
        We update the weights only if 1-|s_i| < tolerance

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
    >>> dW = perceptron_rule_backward(x, y, y_hat, margin, tolerance)
    >>> dW.shape
    (2, 2)

    """
    # regular perceptron rule
    x = jnp.atleast_2d(x)  # (n, d)
    y = jnp.atleast_2d(y)  # (n, K)
    y_hat = jnp.atleast_2d(y_hat)  # (n, K)

    n, d = x.shape
    if y.shape != y_hat.shape or y.shape[0] != n:
        raise ValueError("y and y_hat must have the same (n, K) shape.")
    m = y * y_hat  # (n, K)
    mistake = (m <= margin).astype(x.dtype)  # (n, K)

    # applying gate to saturated neurons
    s = jnp.tanh(y_hat)  # (n, K)
    tol_b = jnp.broadcast_to(tolerance, s.shape)
    gate = ((1.0 - jnp.abs(s)) < tol_b).astype(x.dtype)  # (n, K)
    local = mistake * gate * y  # (n, K)

    # Column-per-class update; normalize by sqrt(n) * sqrt(d)
    dW = (x.T @ local) / (jnp.sqrt(n) * jnp.sqrt(d))  # (d, K)
    return -dW
