import jax
import jax.numpy as jnp
import pytest
from jax.tree_util import tree_leaves

from bionet.modules.ferromagnetic import Ferromagnetic

# ---------- construction ----------


def test_init_with_scalar_strength_broadcasts_to_features():
    """Scalar strength broadcasts to a (features,) vector."""
    features = 5
    m = Ferromagnetic(features=features, strength=0.3)
    assert m.strength.shape == (features,)
    assert jnp.allclose(m.strength, jnp.full((features,), 0.3, dtype=m.strength.dtype))


def test_init_with_vector_strength_correct_length():
    """Vector strength must match the features dimension."""
    vec = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    m = Ferromagnetic(features=3, strength=vec)
    assert m.strength.shape == (3,)
    assert jnp.allclose(m.strength, vec)


def test_init_with_vector_wrong_length_raises():
    """Mismatched strength length raises ValueError."""
    with pytest.raises(ValueError):
        Ferromagnetic(features=4, strength=jnp.ones((3,), dtype=jnp.float32))


def test_init_with_wrong_rank_raises():
    """Higher-rank strength raises ValueError."""
    with pytest.raises(ValueError):
        Ferromagnetic(features=2, strength=jnp.ones((1, 2)))


# ---------- behavior ----------


def test_call_multiplies_elementwise():
    """Forward pass scales inputs elementwise."""
    m = Ferromagnetic(features=3, strength=jnp.array([1.0, 0.5, 2.0], dtype=jnp.float32))
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)  # (2, 3)
    y = m(x)
    expected = x * jnp.array([1.0, 0.5, 2.0], dtype=jnp.float32)
    assert jnp.allclose(y, expected)


def test_has_state_is_false_property():
    """has_state is a property and returns False."""
    m = Ferromagnetic(features=3, strength=1.0)
    assert isinstance(type(m).has_state, property)
    assert m.has_state is False


# ---------- backward/update semantics ----------


def test_backward_returns_zero_update_and_no_mutation():
    """Backward returns a zero update of the same structure without mutation."""
    m = Ferromagnetic(features=3, strength=jnp.array([1.0, 0.5, 2.0], dtype=jnp.float32))
    orig_strength = m.strength.copy()
    update = m.backward(
        x=jnp.ones((3,), dtype=jnp.float32),
        y=jnp.ones((3,), dtype=jnp.float32),
        y_hat=jnp.ones((3,), dtype=jnp.float32),
    )

    m_leaves = tree_leaves(m)
    u_leaves = tree_leaves(update)
    assert len(m_leaves) == len(u_leaves)
    for ml, ul in zip(m_leaves, u_leaves, strict=False):
        assert ul.shape == ml.shape
        assert ul.dtype == ml.dtype
        assert jnp.allclose(ul, jnp.zeros_like(ml))

    # module unchanged
    assert jnp.allclose(m.strength, orig_strength)


# ---------- dtype behavior (optional policy) ----------


def test_call_preserves_input_dtype():
    """Binary op preserves right-hand dtype under JAX rules."""
    m = Ferromagnetic(features=3, strength=1.0, dtype=jnp.float32)
    x = jnp.ones((2, 3), dtype=jnp.float64)
    y = m(x)
    assert y.dtype == x.dtype


# ---------- jittability ----------


def test_jit_call_compiles_and_runs():
    """JIT-compiled forward pass matches eager output."""
    m = Ferromagnetic(features=3, strength=jnp.array([1.0, 0.5, 2.0], dtype=jnp.float32))
    x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    f_jit = jax.jit(lambda mod, t: mod(t))
    y_jit = f_jit(m, x)
    y_eager = m(x)
    assert jnp.allclose(y_jit, y_eager)


def test_jit_backward_compiles_and_runs():
    """JIT-compiled backward returns zero-shaped update matching the module."""
    m = Ferromagnetic(features=4, strength=0.7)
    bwd_jit = jax.jit(lambda mod, x, y, y_hat: mod.backward(x, y, y_hat))
    upd = bwd_jit(
        m,
        x=jnp.ones((3,), dtype=jnp.float32),
        y=jnp.ones((3,), dtype=jnp.float32),
        y_hat=jnp.ones((3,), dtype=jnp.float32),
    )

    m_leaves = tree_leaves(m)
    u_leaves = tree_leaves(upd)
    assert len(m_leaves) == len(u_leaves)
    for ml, ul in zip(m_leaves, u_leaves, strict=True):
        assert ul.shape == ml.shape
        assert jnp.allclose(ul, jnp.zeros_like(ml))
