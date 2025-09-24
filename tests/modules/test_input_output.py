import jax.numpy as jnp

from bionet.modules.input_output import OutputLayer


def test_forward_is_zero():
    """Forward pass simply returns the state."""
    layer = OutputLayer()
    x = jnp.array([1.0, -2.0, 3.5])
    y = layer(x)
    assert jnp.array_equal(y, jnp.zeros_like(y))


def test_activation_is_identity():
    """No activation is performed (Linear)."""
    layer = OutputLayer()
    x = jnp.array([0.1, 0.2])
    a = layer.activation(x)
    assert jnp.array_equal(a, x)


def test_reduce_sums_list_of_arrays():
    """Aggregation is a sum."""
    layer = OutputLayer()
    h = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]
    out = layer.reduce(h)
    assert jnp.allclose(out, jnp.array([4.0, 6.0]))


def test_reduce_sums_nested_pytree():
    """Reduction works on nested pytree."""
    layer = OutputLayer()
    h = {
        "a": jnp.array([1.0, 2.0]),
        "b": (jnp.array([0.0, 0.0]), jnp.array([5.0, 6.0])),
    }
    out = layer.reduce(h)
    assert jnp.allclose(out, jnp.array([6.0, 8.0]))


def test_backward_returns_self():
    """Backward returns itself (no parameters)."""
    layer = OutputLayer()
    x = jnp.array([0.0, 1.0])
    y = jnp.array([0.0, 1.0])
    y_hat = jnp.array([0.5, 0.5])
    assert layer.backward(x, y, y_hat) is layer
