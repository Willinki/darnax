import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves

from bionet.modules.recurrent import RecurrentDiscrete
from bionet.utils.perceptron_rule import perceptron_rule_backward


def test_init_shapes_and_diagonal_set():
    """J is (F,F); diag(J) equals J_D; threshold is (F,)."""
    key = jax.random.key(0)
    F = 4
    j_d = jnp.arange(F, dtype=jnp.float32) * 0.1
    thr = 0.2
    layer = RecurrentDiscrete(features=F, j_d=j_d, threshold=thr, key=key)

    assert layer.J.shape == (F, F)
    assert layer.J_D.shape == (F,)
    assert layer.threshold.shape == (F,)
    assert jnp.allclose(jnp.diag(layer.J), layer.J_D)


def test_activation_is_strict_pm_one_and_ties_to_plus():
    """Activation outputs ±1 only; zero inputs map to +1."""
    key = jax.random.key(1)
    F = 3
    layer = RecurrentDiscrete(F, j_d=0.0, threshold=0.0, key=key)

    x = jnp.array([-2.0, 0.0, 3.5], dtype=jnp.float32)
    a = layer.activation(x)
    assert jnp.array_equal(a, jnp.array([-1.0, 1.0, 1.0], dtype=jnp.float32))
    assert set(jnp.unique(a).tolist()) == {-1.0, 1.0}


def test_forward_shape_and_computation():
    """__call__ returns x @ J with correct shape."""
    key = jax.random.key(2)
    F = 5
    layer = RecurrentDiscrete(F, j_d=0.0, threshold=0.0, key=key)
    x = jax.random.normal(jax.random.key(3), (7, F), dtype=jnp.float32)
    y = layer(x)
    assert y.shape == (7, F)
    # sanity check on one row
    row = x[0]
    assert jnp.allclose(y[0], row @ layer.J)


def test_reduce_sums_messages():
    """Reduce sums a pytree of messages elementwise."""
    key = jax.random.key(4)
    F = 3
    layer = RecurrentDiscrete(F, j_d=0.0, threshold=0.0, key=key)
    msgs = [jnp.ones((2, F), dtype=jnp.float32), 2.0 * jnp.ones((2, F), dtype=jnp.float32)]
    agg = layer.reduce(msgs)
    assert jnp.allclose(agg, 3.0 * jnp.ones((2, F), dtype=jnp.float32))


def test_backward_wraps_deltaj_and_zeros_others():
    """Backward returns module-shaped update: ΔJ set, J_D/thresh zeros."""
    key = jax.random.key(5)
    F = 4
    p_t = 0.5
    layer = RecurrentDiscrete(F, j_d=0.1, threshold=0.2, key=key)

    x = jax.random.normal(jax.random.key(6), (8, F), dtype=jnp.float32)
    y = jnp.where(jax.random.uniform(jax.random.key(7), (8, F)) > p_t, 1.0, -1.0).astype(
        jnp.float32
    )
    y_hat = layer(x)

    upd = layer.backward(x, y, y_hat)

    # same pytree structure
    l_leaves = tree_leaves(layer)
    u_leaves = tree_leaves(upd)
    assert len(l_leaves) == len(u_leaves)

    # ΔJ matches perceptron_rule_backward
    expected_dJ = perceptron_rule_backward(x, y, y_hat, layer.threshold)
    expected_dJ = expected_dJ * (1 - jnp.eye(expected_dJ.shape[0]))
    assert upd.J.shape == layer.J.shape
    assert jnp.allclose(upd.J, expected_dJ)

    # other fields are zero
    assert jnp.allclose(upd.J_D, jnp.zeros_like(layer.J_D))
    assert jnp.allclose(upd.threshold, jnp.zeros_like(layer.threshold))


def test_jit_forward_and_backward():
    """JIT-compiled forward/backward run and match eager outputs structurally."""
    key = jax.random.key(8)
    F = 6
    p_t = 0.5
    layer = RecurrentDiscrete(F, j_d=0.0, threshold=0.0, key=key)

    x = jax.random.normal(jax.random.key(9), (3, F), dtype=jnp.float32)
    y = jnp.where(jax.random.uniform(jax.random.key(10), (3, F)) > p_t, 1.0, -1.0).astype(
        jnp.float32
    )
    y_hat = layer(x)

    fwd_jit = jax.jit(lambda m, t: m(t))
    bwd_jit = jax.jit(lambda m, a, b, c: m.backward(a, b, c))

    y_eager = layer(x)
    y_jit = fwd_jit(layer, x)
    assert jnp.allclose(y_eager, y_jit)

    upd_eager = layer.backward(x, y, y_hat)
    upd_jit = bwd_jit(layer, x, y, y_hat)

    # compare ΔJ and zero-ness of others
    assert jnp.allclose(upd_eager.J, upd_jit.J)
    assert jnp.allclose(upd_eager.J_D, upd_jit.J_D)
    assert jnp.allclose(upd_eager.threshold, upd_jit.threshold)
