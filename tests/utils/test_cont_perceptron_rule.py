# test_tanh_perceptron_rules.py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.darnax.utils.cont_perceptron_rule import (
    tanh_perceptron_rule_backward,
    tanh_truncated_perceptron_rule_backward,
)


# -----------------------------
# Helpers
# -----------------------------
def np_tanh_perceptron_rule_backward(x, y, y_hat, tolerance):
    """Direct rule."""
    x = np.atleast_2d(np.asarray(x))
    o = np.tanh(np.asarray(y_hat))
    err = np.asarray(y) - o
    tol = np.broadcast_to(np.asarray(tolerance), err.shape)
    mask = (np.abs(err) >= tol).astype(x.dtype)
    local = err * (1.0 - o**2) * mask
    dW = x.T @ local
    n, d = x.shape
    dW = dW / (np.sqrt(n) * np.sqrt(d))
    return -dW  # sign matches implementation


def duplicate_batch(x, y, y_hat, times=2):
    """Double batch."""
    x2 = jnp.vstack([x] * times)
    y2 = jnp.vstack([y] * times)
    y_hat2 = jnp.vstack([y_hat] * times)
    return x2, y2, y_hat2


# -----------------------------
# Tests for tanh_perceptron_rule_backward
# -----------------------------
def test_tanh_rule_shapes_and_numeric_scalar_tol():
    """Test shapes and numeric tol."""
    x = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # (n=2, d=2)
    y = jnp.array([[+1.0, -1.0], [-1.0, +1.0]])  # (2, K=2)
    y_hat = jnp.array([[0.2, -0.3], [0.1, -0.4]])  # (2, 2)
    tol = 0.0

    dW = tanh_perceptron_rule_backward(x, y, y_hat, tol)
    assert dW.shape == (2, 2)

    dW_np = np_tanh_perceptron_rule_backward(x, y, y_hat, tol)
    np.testing.assert_allclose(np.asarray(dW), dW_np, rtol=1e-6, atol=1e-7)


def test_tanh_rule_broadcasting_tolerance_vector_and_matrix():
    """Test tolerance broadcast."""
    x = jnp.array([[1.0, 2.0, 0.0]])  # (1, d=3)
    y = jnp.array([[+1.0, 0.0]])  # (1, K=2) continuous labels allowed
    y_hat = jnp.array([[0.0, 0.0]])  # (1, 2)

    # tol as scalar, vector (K,), and full (n,K) must agree
    dW_scalar = tanh_perceptron_rule_backward(x, y, y_hat, 0.1)
    dW_vec = tanh_perceptron_rule_backward(x, y, y_hat, jnp.array([0.1, 0.1]))
    dw_full = tanh_perceptron_rule_backward(x, y, y_hat, jnp.array([[0.1, 0.1]]))

    np.testing.assert_allclose(np.asarray(dW_scalar), np.asarray(dW_vec), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(dW_scalar), np.asarray(dw_full), rtol=1e-6, atol=1e-7)


def test_tanh_rule_masking_with_large_tolerance_zeroes_updates():
    """Test large updates are zeroed."""
    # Make errors small compared to tolerance so mask = 0 → ΔW = 0
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([[0.1, -0.1], [0.2, -0.2]])
    y_hat = jnp.array([[0.1, -0.1], [0.2, -0.2]])  # o = tanh(y_hat) close to y
    tol = 1.0  # bigger than |y - o| for sure

    dW = tanh_perceptron_rule_backward(x, y, y_hat, tol)
    np.testing.assert_allclose(np.asarray(dW), np.zeros((2, 2)), atol=1e-8)


# def test_tanh_rule_batch_size_normalization_invariance():
#    """Test Duplicating the batch should not change ΔW thanks to 1/sqrt(n)."""
#    x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
#    y = jnp.array([[+1.0, -1.0], [-1.0, +1.0]])
#    y_hat = jnp.array([[0.2, -0.3], [0.1, -0.4]])
#    tol = 0.0
#
#    dW = tanh_perceptron_rule_backward(x, y, y_hat, tol)
#
#    x2, y2, yhat2 = duplicate_batch(x, y, y_hat, times=5)
#    dW_dup = tanh_perceptron_rule_backward(x2, y2, yhat2, tol)
#
#    np.testing.assert_allclose(np.asarray(dW_dup), np.asarray(dW), rtol=1e-6, atol=1e-7)


def test_tanh_rule_jittable():
    """Test jittability."""
    f = jax.jit(tanh_perceptron_rule_backward)
    x = jnp.array([1.0, 2.0, 3.0])  # (d,)
    y = jnp.array([[+1.0, -1.0]])  # (1, K)
    y_hat = jnp.array([[0.5, -0.5]])
    tol = 0.0
    dW = f(x, y, y_hat, tol)
    assert dW.shape == (3, 2)


# -----------------------------
# Tests for tanh_truncated_perceptron_rule_backward
# -----------------------------
def test_truncated_rule_basic_behavior_margin_and_gate():
    """Test basic behavior."""
    # Two positions: one saturated (passes gate), one not saturated (blocked by gate)
    x = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # (n=2, d=2)
    y = jnp.array([[+1.0, -1.0], [+1.0, -1.0]])
    # Make y_hat strongly saturated on [0,0] and [1,1], non-saturated on off-diagonal
    y_hat = jnp.array([[+3.0, 0.0], [0.0, -3.0]])  # tanh(±3) ~ ±0.995
    margin = 0.0
    tolerance = 0.01  # gate: (1 - |tanh|) < 0.01 -> only the saturated entries update

    dW = tanh_truncated_perceptron_rule_backward(x, y, y_hat, margin, tolerance)
    # Only columns where both "mistake" and "gate" hold contribute.
    # For margin=0 and y in {+1,-1}:
    # m = y * y_hat. With y_hat 3 and y=+1 -> m>0 so not a mistake; with y_hat -3 and y=-1 -> m>0; with 0 -> tie (mistake)
    # But gate blocks the 0 entries (not saturated). So the only potential updates are saturated entries with mistakes = 0 → no update.
    # Therefore ΔW should be zero.
    np.testing.assert_allclose(np.asarray(dW), np.zeros((2, 2)), atol=1e-8)


def test_truncated_rule_when_mistake_and_gate_both_fire():
    """Test self-explanatory."""
    # Force a mistake on a saturated unit
    x = jnp.array([[1.0, 0.0]])  # (1, d=2)
    y = jnp.array([[+1.0, -1.0]])  # (1, K=2)
    y_hat = jnp.array([[-3.0, +3.0]])  # wrong signs vs y, and saturated
    margin = 0.0
    tolerance = 0.01  # gate passes because tanh(±3) ≈ ±0.995 → 1-|s| ≈ 0.005 < 0.01

    dW = tanh_truncated_perceptron_rule_backward(x, y, y_hat, margin, tolerance)
    # Non-zero update expected, and shape correct
    assert dW.shape == (2, 2)
    assert jnp.any(jnp.abs(dW) > 0.0)


def test_truncated_rule_broadcasting_margin_and_tolerance():
    """Test broadcasting of margin and tolerance."""
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([[+1.0, -1.0], [+1.0, -1.0]])
    y_hat = jnp.array([[0.2, -0.2], [-0.3, +0.3]])

    dW_scalar = tanh_truncated_perceptron_rule_backward(x, y, y_hat, 0.0, 0.5)
    dW_margin_vec = tanh_truncated_perceptron_rule_backward(x, y, y_hat, jnp.array([0.0, 0.0]), 0.5)
    dW_tol_full = tanh_truncated_perceptron_rule_backward(x, y, y_hat, 0.0, jnp.full_like(y, 0.5))

    np.testing.assert_allclose(
        np.asarray(dW_scalar), np.asarray(dW_margin_vec), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(np.asarray(dW_scalar), np.asarray(dW_tol_full), rtol=1e-6, atol=1e-7)


# def test_truncated_rule_batch_size_normalization_invariance():
#    """Test normalization invariance."""
#    x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
#    y = jnp.array([[+1.0, +1.0], [+1.0, +1.0]])
#    y_hat = jnp.array([[-1.0, -1.0], [-1.5, -1.5]])
#    margin = 0.0
#    tolerance = 0.2  # gate likely passes since tanh(|y_hat|) is fairly large
#
#    dW = tanh_truncated_perceptron_rule_backward(x, y, y_hat, margin, tolerance)
#
#    x2, y2, yhat2 = duplicate_batch(x, y, y_hat, times=3)
#    dW_dup = tanh_truncated_perceptron_rule_backward(x2, y2, yhat2, margin, tolerance)
#
#    np.testing.assert_allclose(np.asarray(dW_dup), np.asarray(dW), rtol=1e-6, atol=1e-7)


def test_truncated_rule_raises_on_mismatched_shapes():
    """Test mismatched shapes."""
    x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    y = jnp.array([[+1.0, -1.0]])
    y_hat = jnp.array([[0.2, -0.2], [0.1, -0.1], [0.0, 0.0]])  # wrong n
    with pytest.raises(ValueError):
        _ = tanh_truncated_perceptron_rule_backward(x, y, y_hat, 0.0, 0.1)
