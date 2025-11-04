# test_recurrent_tanh.py
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.darnax.modules.recurrent_tanh import RecurrentTanh, RecurrentTanhTruncated


def offdiag(m):
    """Remove off-diag terms."""
    d = m.shape[0]
    return m * (1 - jnp.eye(d, dtype=m.dtype))


def test_init_shapes_and_diagonal_and_mask():
    """Test shapes."""
    key = jax.random.PRNGKey(0)
    d = 5
    j_d = 0.3
    tol = 0.1
    layer = RecurrentTanh(features=d, j_d=j_d, tolerance=tol, key=key, strength=1.0)

    assert layer.J.shape == (d, d)
    np.testing.assert_allclose(jnp.diag(layer.J), jnp.full((d,), j_d), atol=1e-7)

    # mask zeros the diagonal and keeps off-diagonals as 1
    np.testing.assert_allclose(jnp.diag(layer._mask), jnp.zeros((d,)), atol=0)
    np.testing.assert_allclose(offdiag(layer._mask), offdiag(jnp.ones((d, d))), atol=0)

    # sanity: strength is scalar
    assert jnp.isscalar(layer.strength)


def test_strength_scales_offdiagonals_with_same_seed():
    """Test strength scales."""
    key = jax.random.PRNGKey(42)
    d = 6
    j_d = 0.0
    tol = 0.0

    layer1 = RecurrentTanh(d, j_d, tol, key, strength=1.0)
    # reuse same key to compare deterministic off-diagonals
    layer2 = RecurrentTanh(d, j_d, tol, key, strength=2.0)

    off1 = offdiag(layer1.J)
    off2 = offdiag(layer2.J)
    # Diagonals are forced to j_d; ignore them. Off-diagonals should scale ~2x
    np.testing.assert_allclose(off2, off1 * 2.0, rtol=1e-6, atol=1e-7)


def test_activation_and_forward_shapes():
    """Test activation and shapes."""
    key = jax.random.PRNGKey(1)
    d = 4
    layer = RecurrentTanh(d, j_d=-0.2, tolerance=0.05, key=key)

    # vector
    x = jnp.arange(d, dtype=jnp.float32)
    h = layer(x)
    assert h.shape == (d,)

    # batch
    xb = jnp.stack([x, x + 1.0], axis=0)
    hb = layer(xb)
    assert hb.shape == (2, d)

    # activation is tanh and preserves dtype/shape
    a = layer.activation(hb)
    assert a.shape == (2, d)
    assert a.dtype == hb.dtype
    assert jnp.all((a >= -1) & (a <= 1))


def test_reduce_sums_pytree_leaves():
    """Test sum reduction."""
    key = jax.random.PRNGKey(2)
    d = 3
    layer = RecurrentTanh(d, j_d=0.0, tolerance=0.0, key=key)

    h = {"a": jnp.ones((d,)), "b": jnp.full((d,), 2.0)}
    r = layer.reduce(h)
    np.testing.assert_allclose(r, jnp.full((d,), 3.0), atol=1e-7)


def test_backward_broadcasting_tolerance_equivalence():
    """Test backward broadcasting."""
    key = jax.random.PRNGKey(4)
    d = 4
    x = jnp.ones((2, d), dtype=jnp.float32)
    y = jnp.ones((2, d))
    y_hat = jnp.zeros((2, d))

    layer_s = RecurrentTanh(d, j_d=0.0, tolerance=0.2, key=key)
    # reinit with same key to keep J identical
    layer_v = RecurrentTanh(d, j_d=0.0, tolerance=jnp.full((d,), 0.2), key=key)

    upd_s = layer_s.backward(x, y, y_hat)
    upd_v = layer_v.backward(x, y, y_hat)
    np.testing.assert_allclose(upd_s.J, upd_v.J, rtol=1e-6, atol=1e-7)


def test_jittable_forward_and_backward():
    """Test jittability."""
    key = jax.random.PRNGKey(5)
    d = 3
    layer = RecurrentTanh(d, j_d=0.0, tolerance=0.0, key=key)

    x = jnp.ones((d,), dtype=jnp.float32)
    y = jnp.ones((1, d))
    y_hat = jnp.zeros((1, d))

    # Forward via eqx.filter_jit
    fwd = eqx.filter_jit(lambda m, z: m(z))
    h = fwd(layer, x)
    assert h.shape == (d,)

    # Backward via eqx.filter_jit
    bwd = eqx.filter_jit(lambda m, xb, yb, yh: m.backward(xb, yb, yh))
    upd = bwd(layer, x[None, :], y, y_hat)
    assert upd.J.shape == (d, d)


def test__set_shape_valid_and_errors():
    """Test set_shape."""
    key = jax.random.PRNGKey(6)
    d = 4
    layer = RecurrentTanh(d, j_d=0.0, tolerance=0.0, key=key)

    v = layer._set_shape(0.5, d, jnp.float32)
    assert v.shape == (d,)

    v2 = layer._set_shape(jnp.arange(d, dtype=jnp.float32), d, jnp.float32)
    assert v2.shape == (d,)

    with pytest.raises(ValueError):
        _ = layer._set_shape(jnp.arange(d + 1), d, jnp.float32)
    with pytest.raises(ValueError):
        _ = layer._set_shape(jnp.ones((2, 2)), d, jnp.float32)


# ---------------------------
# RecurrentTanhTruncated
# ---------------------------


def test_truncated_init_and_backward_nonzero_and_masked():
    """Test truncated."""
    key = jax.random.PRNGKey(7)
    d = 3
    # margin (threshold) = 0, tolerance small; choose y_hat to be saturated with wrong sign
    layer = RecurrentTanhTruncated(
        features=d,
        j_d=0.0,
        tolerance=0.01,
        threshold=0.0,  # margin
        key=key,
    )

    x = jnp.eye(d, dtype=jnp.float32)
    y = jnp.array([[+1.0, -1.0, +1.0]] * d)
    y_hat = jnp.array([[-3.0, +3.0, -3.0]] * d)  # wrong sign + saturated → mistake & gate fire

    upd = layer.backward(x, y, y_hat)
    assert upd.J.shape == (d, d)
    assert jnp.any(jnp.abs(offdiag(upd.J)) > 0.0)
    # diagonal must be zero due to mask
    np.testing.assert_allclose(jnp.diag(upd.J), jnp.zeros((d,)), atol=1e-8)


def test_truncated_broadcasting_threshold_and_tolerance():
    """Test broadcasting."""
    key = jax.random.PRNGKey(8)
    d = 4
    x = jnp.ones((2, d), dtype=jnp.float32)
    y = jnp.ones((2, d))
    y_hat = jnp.zeros((2, d))

    # Same seed so J is identical
    layer_a = RecurrentTanhTruncated(d, j_d=0.0, tolerance=0.2, threshold=0.0, key=key)
    layer_b = RecurrentTanhTruncated(
        d, j_d=0.0, tolerance=jnp.full((d,), 0.2), threshold=jnp.zeros((d,)), key=key
    )

    ua = layer_a.backward(x, y, y_hat)
    ub = layer_b.backward(x, y, y_hat)
    np.testing.assert_allclose(ua.J, ub.J, rtol=1e-6, atol=1e-7)


# ---------------------------
# Contract checks / gotchas
# ---------------------------


def test_api_contract_exposes_tolerance_and_keeps_diag_fixed():
    """Test will FAIL with the current code if `RecurrentTanh.__init__`.

    stores `threshold` instead of `tolerance`. That’s on purpose to flag the bug.
    """
    key = jax.random.PRNGKey(9)
    d = 3
    tol = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    j_d = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32)

    layer = RecurrentTanh(d, j_d=j_d, tolerance=tol, key=key)
    # Expect the attribute to exist and match
    np.testing.assert_allclose(layer.tolerance, tol, atol=0)

    # And diag(J) must equal J_D throughout usage
    np.testing.assert_allclose(jnp.diag(layer.J), layer.J_D, atol=1e-7)
