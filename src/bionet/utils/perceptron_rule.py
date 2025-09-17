"""Perceptron (OVA) backward update in JAX.

- Supports batched inputs.
- Labels are {-1, +1} per class (one-vs-all).
- `y_hat` are raw scores (pre-activation).
- Returns weight update `dW` without applying a learning rate.

Shapes
------
x      : (d,) or (n, d)
y      : (n, K) in {-1, +1}
y_hat  : (n, K) raw scores
returns: (d, K)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def perceptron_rule_backward(
    x: jax.Array,
    y: jax.Array,
    y_hat: jax.Array,
    margin: jax.Array,
) -> jax.Array:
    """Compute multiclass (OVA) perceptron weight update (no learning rate).

    Updates are applied when ``y * y_hat <= margin`` (ties counted as mistakes).

    Args:
        x: Input vector(s), shape ``(d,)`` or ``(n, d)``.
        y: Targets in ``{-1, +1}``, shape ``(n, K)``.
        y_hat: Raw scores, shape ``(n, K)``.
        margin: Scalar margin threshold; changing its value will not retrace.

    Returns:
        Weight update ``dW`` of shape ``(d, K)`` to add to the weights.

    """
    x = jnp.atleast_2d(x)  # (n, d)
    y = jnp.atleast_2d(y)  # (n, K)
    y_hat = jnp.atleast_2d(y_hat)  # (n, K)

    n = x.shape[0]
    if y.shape != y_hat.shape or y.shape[0] != n:
        raise ValueError("y and y_hat must have the same (n, K) shape.")
    m = y * y_hat  # (n, K)
    mistake = (m <= margin).astype(x.dtype)  # (n, K)
    update = x.T @ (mistake * y)
    return update
