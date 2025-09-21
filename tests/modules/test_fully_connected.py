import jax
import jax.numpy as jnp
import pytest
from jax.tree_util import tree_leaves

from bionet.modules.fully_connected import FrozenFullyConnected, FullyConnected
from bionet.utils.perceptron_rule import perceptron_rule_backward

PRECISION = 0.5
# ---------- construction ----------


def test_init_shapes_and_types():
    """W is (in,out); strength/threshold are (out,)."""
    key = jax.random.key(0)
    m = FullyConnected(3, 5, strength=0.5, threshold=0.2, key=key)
    assert m.W.shape == (3, 5)
    assert m.strength.shape == (5,)
    assert m.threshold.shape == (5,)


def test_init_with_vector_params_and_length_check():
    """Vector strength/threshold must match out_features."""
    key = jax.random.key(1)
    strength = jnp.arange(4, dtype=jnp.float32) * 0.1
    thr = jnp.ones((4,), dtype=jnp.float32) * 0.2
    m = FullyConnected(3, 4, strength=strength, threshold=thr, key=key)
    assert jnp.allclose(m.strength, strength)
    assert jnp.allclose(m.threshold, thr)

    with pytest.raises(ValueError):
        FullyConnected(3, 4, strength=jnp.ones((3,), jnp.float32), threshold=0.2, key=key)

    with pytest.raises(ValueError):
        FullyConnected(3, 4, strength=0.5, threshold=jnp.ones((3,), jnp.float32), key=key)


def test_init_with_2d_vector():
    """Raise value error if strength is a 2d vector."""
    key = jax.random.key(1)
    with pytest.raises(ValueError):
        FullyConnected(3, 4, strength=jnp.ones((3, 3), jnp.float32), threshold=0.2, key=key)


# ---------- behavior ----------


def test_forward_broadcasts_strength_over_batch():
    """(x @ W) * strength matches explicit broadcasting."""
    key = jax.random.key(2)
    m = FullyConnected(
        4, 3, strength=jnp.array([1.0, 0.5, 2.0], jnp.float32), threshold=0.0, key=key
    )
    x = jax.random.normal(jax.random.key(3), (7, 4), dtype=jnp.float32)
    y = m(x)
    expected = (x @ m.W) * m.strength
    assert jnp.allclose(y, expected)


def test_has_state_is_false_property():
    """Adapters report has_state == False (even if trainable)."""
    m = FullyConnected(2, 2, strength=1.0, threshold=0.0, key=jax.random.key(4))
    assert isinstance(FullyConnected.has_state, property)
    assert m.has_state is False


# ---------- backward/update semantics ----------


def test_backward_sets_only_w_and_zeros_others():
    """Update tree has ΔW and zeros for strength/threshold."""
    key = jax.random.key(5)
    in_f, out_f, n = 3, 4, 8
    m = FullyConnected(in_f, out_f, strength=1.0, threshold=0.1, key=key)

    x = jax.random.normal(jax.random.key(6), (n, in_f), dtype=jnp.float32)
    y = jnp.where(jax.random.uniform(jax.random.key(7), (n, out_f)) > PRECISION, 1.0, -1.0).astype(
        jnp.float32
    )
    y_hat = m(x)

    upd = m.backward(x, y, y_hat)
    expected_dW = perceptron_rule_backward(x, y, y_hat, m.threshold)

    # ΔW equals rule output
    assert upd.W.shape == m.W.shape
    assert jnp.allclose(upd.W, expected_dW)

    # other leaves zero
    assert jnp.allclose(upd.strength, jnp.zeros_like(m.strength))
    assert jnp.allclose(upd.threshold, jnp.zeros_like(m.threshold))

    # structure preserved
    ml, ul = tree_leaves(m), tree_leaves(upd)
    assert len(ml) == len(ul)


# ---------- jittability ----------


def test_jit_forward_and_backward():
    """JIT-compiled forward/backward run and match eager results structurally."""
    key = jax.random.key(8)
    in_f, out_f, n = 5, 6, 4
    m = FullyConnected(in_f, out_f, strength=1.0, threshold=0.0, key=key)

    x = jax.random.normal(jax.random.key(9), (n, in_f), dtype=jnp.float32)
    y = jnp.where(jax.random.uniform(jax.random.key(10), (n, out_f)) > PRECISION, 1.0, -1.0).astype(
        jnp.float32
    )
    y_hat = m(x)

    fwd_jit = jax.jit(lambda mod, t: mod(t))
    bwd_jit = jax.jit(lambda mod, a, b, c: mod.backward(a, b, c))

    y_eager, y_j = m(x), fwd_jit(m, x)
    assert jnp.allclose(y_eager, y_j)

    upd_eager = m.backward(x, y, y_hat)
    upd_j = bwd_jit(m, x, y, y_hat)
    assert jnp.allclose(upd_eager.W, upd_j.W)
    assert jnp.allclose(upd_eager.strength, upd_j.strength)
    assert jnp.allclose(upd_eager.threshold, upd_j.threshold)


# ---------- frozen fully connected ----------


def test_frozen_and_fully_connected_return_the_same():
    """Test that when the regular and frozen modules are equivalent.

    We do this by initializing the weights with the same seed and check
    for equal prediction.
    """
    key = jax.random.key(8)
    in_f, out_f, n = 5, 6, 4
    # init with same key, on purpose
    m = FullyConnected(in_f, out_f, strength=1.0, threshold=0.0, key=key)
    m_frozen = FrozenFullyConnected(in_f, out_f, strength=1.0, threshold=0.0, key=key)

    x = jax.random.normal(jax.random.key(9), (n, in_f), dtype=jnp.float32)

    y_regular, y_frozen = m(x), m_frozen(x)
    assert jnp.allclose(y_regular, y_frozen)


def test_frozen_does_not_update():
    """Check that all parameters are fixed."""
    key = jax.random.key(8)
    in_f, out_f, n = 5, 6, 4
    m_frozen = FrozenFullyConnected(in_f, out_f, strength=1.0, threshold=0.0, key=key)

    x = jax.random.normal(jax.random.key(9), (n, in_f), dtype=jnp.float32)
    y = jnp.where(jax.random.uniform(jax.random.key(10), (n, out_f)) > PRECISION, 1.0, -1.0).astype(
        jnp.float32
    )
    y_hat = m_frozen(x)

    updates = m_frozen.backward(x, y, y_hat)

    assert jnp.allclose(updates.W, 0), updates.W
    assert jnp.allclose(updates.strength, 0)
    assert jnp.allclose(updates.threshold, 0)
