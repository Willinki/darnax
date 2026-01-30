import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import pytest

from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete, Conv2DTranspose


def _rand(key, shape, dtype=jnp.float32):
    return jax.random.normal(key, shape, dtype=dtype)


def _assert_tree_allclose(a, b, *, atol=0.0, rtol=0.0):
    """Compare two pytrees leafwise for arrays; non-arrays are compared by ==."""
    la = jtu.tree_leaves(a)
    lb = jtu.tree_leaves(b)
    assert len(la) == len(lb)
    for xa, xb in zip(la, lb):
        if isinstance(xa, jax.Array) or hasattr(xa, "dtype"):
            assert jnp.allclose(xa, xb, atol=atol, rtol=rtol)
        else:
            assert xa == xb


@pytest.mark.parametrize(
    "module_factory, x_shape",
    [
        (
            lambda key: Conv2D(
                in_channels=3,
                out_channels=5,
                kernel_size=(3, 3),
                threshold=0.1,
                strength=1.0,
                key=key,
                stride=(1, 1),
                padding_mode="constant",
            ),
            (2, 8, 9, 3),
        ),
        (
            lambda key: Conv2DTranspose(
                in_channels=2,
                out_channels=3,
                kernel_size=(3, 3),
                threshold=0.0,
                strength=1.0,
                key=key,
                stride=2,
                padding_mode="constant",
            ),
            (2, 5, 6, 2),
        ),
        (
            lambda key: Conv2DRecurrentDiscrete(
                channels=6,
                kernel_size=(3, 3),
                groups=3,
                j_d=1.5,
                threshold=0.0,
                key=key,
                padding_mode="constant",
            ),
            (2, 7, 7, 6),
        ),
    ],
)
def test_conv_modules_are_pytree_and_partitionable(module_factory, x_shape):
    key = jax.random.PRNGKey(0)
    mod = module_factory(key)
    x = _rand(jax.random.PRNGKey(1), x_shape)

    # 1) pytree flatten/unflatten
    leaves, treedef = jtu.tree_flatten(mod)
    mod2 = jtu.tree_unflatten(treedef, leaves)
    assert type(mod2) is type(mod)

    # 2) eqx.partition + eqx.combine roundtrip
    params, static = eqx.partition(mod, eqx.is_inexact_array)
    mod3 = eqx.combine(params, static)
    assert type(mod3) is type(mod)

    # roundtrip should preserve outputs
    y1 = mod(x)
    y3 = mod3(x)
    assert y1.shape == y3.shape
    assert jnp.allclose(y1, y3)

    # 3) filter_jit works (equinoxability under JIT)
    y_jit = eqx.filter_jit(lambda m, a: m(a))(mod, x)
    assert y_jit.shape == y1.shape
    assert jnp.allclose(y_jit, y1)


def test_conv2d_backward_is_partitionable_and_jittable():
    key = jax.random.PRNGKey(2)
    kx, km = jax.random.split(key)
    mod = Conv2D(
        in_channels=4,
        out_channels=6,
        kernel_size=(3, 3),
        threshold=0.1,
        strength=1.0,
        key=km,
        stride=(1, 1),
        padding_mode="constant",
    )
    x = _rand(kx, (2, 8, 8, 4))
    y = mod(x)
    y_hat = jnp.zeros_like(y)

    upd = mod.backward(x, y, y_hat, gate=None)
    assert isinstance(upd, Conv2D)
    assert upd.kernel.shape == mod.kernel.shape
    assert jnp.isfinite(upd.kernel).all()

    # update must be partitionable too
    p, s = eqx.partition(upd, eqx.is_inexact_array)
    upd2 = eqx.combine(p, s)
    assert isinstance(upd2, Conv2D)
    assert upd2.kernel.shape == mod.kernel.shape

    # backward should be jittable
    upd_jit = eqx.filter_jit(lambda m, a, b, c: m.backward(a, b, c, None))(mod, x, y, y_hat)
    assert upd_jit.kernel.shape == mod.kernel.shape
    assert jnp.isfinite(upd_jit.kernel).all()


def test_conv2drecurrentdiscrete_backward_is_partitionable_and_respects_constraint():
    key = jax.random.PRNGKey(3)
    mod = Conv2DRecurrentDiscrete(
        channels=6,
        kernel_size=(3, 3),
        groups=3,
        j_d=2.0,
        threshold=0.0,
        key=key,
        padding_mode="constant",
    )
    x = _rand(jax.random.PRNGKey(4), (2, 9, 9, 6))
    y = mod(x)
    y_hat = jnp.zeros_like(y)

    upd = mod.backward(x, y, y_hat, gate=None)
    assert isinstance(upd, Conv2DRecurrentDiscrete)
    assert upd.kernel.shape == mod.kernel.shape

    # constrained center diagonal entries should be zero in the update
    ch, cw = mod.central_element
    mask = mod._central_diag_mask()
    assert jnp.allclose(upd.kernel[ch, cw, :, :] * mask, 0.0)

    # partitionable + jittable backward
    _ = eqx.partition(upd, eqx.is_inexact_array)
    upd_jit = eqx.filter_jit(lambda m, a, b, c: m.backward(a, b, c, None))(mod, x, y, y_hat)
    assert jnp.allclose(upd_jit.kernel[ch, cw, :, :] * mask, 0.0)


def test_conv2dtranspose_backward_is_partitionable_and_jittable():
    key = jax.random.PRNGKey(5)
    kx, km = jax.random.split(key)

    mod = Conv2DTranspose(
        in_channels=2,
        out_channels=3,
        kernel_size=(3, 3),
        threshold=0.0,
        strength=1.0,
        key=km,
        stride=2,
        padding_mode="constant",
    )
    x = _rand(kx, (2, 5, 6, 2))
    y = mod(x)
    y_hat = jnp.zeros_like(y)

    upd = mod.backward(x, y, y_hat, gate=None)
    assert isinstance(upd, Conv2DTranspose)
    assert upd.kernel.shape == mod.kernel.shape
    assert jnp.isfinite(upd.kernel).all()

    # partitionable
    p, s = eqx.partition(upd, eqx.is_inexact_array)
    upd2 = eqx.combine(p, s)
    assert isinstance(upd2, Conv2DTranspose)
    assert upd2.kernel.shape == mod.kernel.shape

    # jittable backward
    upd_jit = eqx.filter_jit(lambda m, a, b, c: m.backward(a, b, c, None))(mod, x, y, y_hat)
    assert upd_jit.kernel.shape == mod.kernel.shape
    assert jnp.isfinite(upd_jit.kernel).all()
