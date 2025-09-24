"""Tests for the multiclass (OVA) perceptron backward rule."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bionet.utils.perceptron_rule import perceptron_rule_backward


def _ref_update(x: jax.Array, y: jax.Array, y_hat: jax.Array, margin) -> jax.Array:
    """Mirror the function's docstring semantics."""
    n = x.shape[0]
    x = jnp.atleast_2d(x)  # (n, d)
    y = jnp.atleast_2d(y)  # (n, K)
    y_hat = jnp.atleast_2d(y_hat)  # (n, K)
    mistake = (y * y_hat <= margin).astype(x.dtype)  # (n, K)
    return -x.T @ (mistake * y) / (n)  # (d, K)


def test_single_example_identity_shapes_and_values():
    """For a single (d,), (K,) example, update equals x ⊗ (mistake * y)."""
    x = jnp.array([1.0, 2.0, 3.0])  # (d,)
    y = jnp.array([+1.0, -1.0])  # (K,)
    y_hat = jnp.array([0.4, 0.9])  # (K,)
    margin = 0.5

    # y * y_hat = [0.4, -0.9] → both <= 0.5 → mistakes both True
    out = perceptron_rule_backward(x, y, y_hat, margin)
    expected = -jnp.array([[+1.0, -1.0], [+2.0, -2.0], [+3.0, -3.0]])
    assert out.shape == (3, 2)
    assert jnp.allclose(out, expected)


@pytest.mark.parametrize(
    "n,d,k,margin",
    [
        (1, 3, 2, 0.0),
        (4, 5, 3, 0.25),
        (8, 2, 1, 0.5),
    ],
)
def test_batched_matches_reference(n: int, d: int, k: int, margin: float):
    """Batched updates match a straightforward reference computation."""
    key = jax.random.key(0)
    kx, ky, ks = jax.random.split(key, 3)
    p = 0.5

    x = jax.random.normal(kx, (n, d))
    # y in {-1, +1}
    y = jnp.where(jax.random.uniform(ky, (n, k)) > p, 1.0, -1.0)
    y_hat = jax.random.normal(ks, (n, k))

    out = perceptron_rule_backward(x, y, y_hat, margin)
    ref = _ref_update(x, y, y_hat, margin)
    assert out.shape == (d, k)
    assert jnp.allclose(out, ref)


def test_ties_count_as_mistakes():
    """When y * y_hat == margin, the update is applied (<=, not <)."""
    x = jnp.array([2.0, -1.0])  # (d,)
    y = jnp.array([+1.0, -1.0])  # (K,)
    # Choose y_hat so that y * y_hat == margin for both classes
    margin = 0.3
    y_hat = jnp.array([0.3, +0.3])  # y*y_hat = [0.3, -0.3] == +/- margin
    out = perceptron_rule_backward(x, y, y_hat, margin)
    # both counted as mistakes → x ⊗ [ +1, -1 ]
    expected = -jnp.array([[+2.0, -2.0], [-1.0, +1.0]])
    assert jnp.allclose(out, expected)


@pytest.mark.parametrize(
    "margin,expect_fn",
    [
        (jnp.inf, lambda x, y: x.T @ y),  # always mistakes → X^T @ y
        (-jnp.inf, lambda x, y: jnp.zeros((x.shape[1], y.shape[1]))),  # never mistakes → zeros
    ],
)
def test_margin_extremes(margin, expect_fn):
    """Extreme margins recover intuitive limits (always vs never update)."""
    x = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # (n=2, d=2)
    y = jnp.array([[+1.0, -1.0], [-1.0, +1.0]])  # (n=2, K=2)
    y_hat = jnp.zeros_like(y)  # values irrelevant at extremes

    out = perceptron_rule_backward(x, y, y_hat, margin)
    expected = -expect_fn(x, y) / x.shape[0]
    assert jnp.allclose(out, expected), f"{out=} - {expected=}"


def test_batch_equals_mean_of_singletons():
    """Linearity: f(X, Y, S) equals sum_i f(x_i, y_i, s_i)."""
    key = jax.random.key(123)
    kx, ky, ks = jax.random.split(key, 3)
    n, d, K = 3, 4, 2
    margin = 0.1
    p = 0.5

    X = jax.random.normal(kx, (n, d))
    Y = jnp.where(jax.random.uniform(ky, (n, K)) > p, 1.0, -1.0)
    S = jax.random.normal(ks, (n, K))

    batched = perceptron_rule_backward(X, Y, S, margin)

    singles = (
        sum(
            (perceptron_rule_backward(X[i], Y[i], S[i], margin) for i in range(n)),
            start=jnp.zeros_like(batched),
        )
        / X.shape[0]
    )
    assert jnp.allclose(batched, singles)


@pytest.mark.parametrize(
    "y_shape,yhat_shape",
    [
        ((2, 3), (2, 2)),  # mismatched K
        ((3, 2), (2, 2)),  # mismatched n
    ],
)
def test_shape_mismatch_raises(y_shape, yhat_shape):
    """The rule validates that y and y_hat share the same (n, K) shape."""
    x = jnp.ones((2, 5))
    y = jnp.ones(y_shape)
    y_hat = jnp.ones(yhat_shape)
    with pytest.raises(ValueError):
        perceptron_rule_backward(x, y, y_hat, margin=0.0)


def test_output_dtype_and_shape():
    """Output has shape (d, K) and inherits dtype from x."""
    x = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)  # (n=2, d=2)
    y = jnp.array([[1, -1], [1, -1]], dtype=jnp.float32)  # (n=2, K=2)
    y_hat = jnp.zeros_like(y)
    out = perceptron_rule_backward(x, y, y_hat, margin=0.0)
    assert out.shape == (2, 2)
    assert out.dtype == x.dtype
