import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from darnax.modules.conv.utils import pad_2d

from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete, Conv2DTranspose
from darnax.modules.conv.utils import conv_forward, conv_transpose_forward


def _rand(key, shape, dtype=jnp.float32):
    return jax.random.normal(key, shape, dtype=dtype)


def test_conv2d_forward_shape_and_jittable():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    x = _rand(k1, (2, 8, 10, 3))
    # kernel shape (Kh, Kw, Cin, Cout)
    conv = Conv2D(
        in_channels=3,
        out_channels=5,
        kernel_size=(3, 3),
        threshold=0.0,
        strength=1.0,
        key=k2,
        stride=(1, 1),
        padding_mode="constant",
    )
    # compute forward both via module and via conv_forward to assert compatibility
    y_module = conv(x)
    y_ref = conv_forward(x, conv.kernel, stride=(1, 1), padding_mode="constant")
    assert y_module.shape == y_ref.shape
    assert jnp.allclose(y_module, y_ref)

    # jitted call (module passed as arg)
    jit_call = jax.jit(lambda m, inp: m(inp))
    y_jit = jit_call(conv, x)
    assert jnp.allclose(y_jit, y_ref)


def test_conv2d_backward_returns_kernel_shaped_update_and_jittable():
    key = jax.random.PRNGKey(1)
    kx, kw = jax.random.split(key)
    x = _rand(kx, (2, 7, 9, 4))
    conv = Conv2D(
        in_channels=4,
        out_channels=6,
        kernel_size=(3, 3),
        threshold=0.1,
        strength=0.5,
        key=kw,
        stride=(1, 1),
        padding_mode="constant",
    )

    # fake forward / supervision
    y = conv(x)
    y_hat = jnp.zeros_like(y)

    upd = conv.backward(x, y, y_hat, gate=None)
    # backward should return an object of same type (Adapter subclass) and a pytree
    assert isinstance(upd, Conv2D)

    # its kernel field should have same shape as the conv.kernel
    assert hasattr(upd, "kernel")
    assert upd.kernel.shape == conv.kernel.shape

    # jitted backward (module passed as captured arg) should run
    jit_bwd = jax.jit(lambda module, a, b, c: module.backward(a, b, c, None))
    upd_jit = jit_bwd(conv, x, y, y_hat)
    assert isinstance(upd_jit, Conv2D)
    assert upd_jit.kernel.shape == conv.kernel.shape


def test_conv2drecurrentdiscrete_projection_and_forward_jd_effect():
    # small channels so we can inspect masks easily
    key = jax.random.PRNGKey(2)
    channels = 4
    groups = 2
    kh, kw = (3, 3)
    j_d = 2.5
    conv = Conv2DRecurrentDiscrete(
        channels=channels,
        kernel_size=(kh, kw),
        groups=groups,
        j_d=j_d,
        threshold=0.0,
        key=key,
        padding_mode="constant",
    )

    # stored kernel central positions must be projected to zero (non-learnable)
    ch, cw = conv.central_element
    # stored kernel has shape (kh, kw, cin_g, cout)
    stored_center = conv.kernel[ch, cw, :, :]
    # there must be zeros where the mask places the constrained diagonal
    mask = conv._central_diag_mask()
    # constrained positions in stored_center should be zero
    constrained_vals = stored_center * mask
    assert jnp.allclose(constrained_vals, 0.0)

    # Now check forward uses j_d at those central positions:
    # construct dummy input and compute forward output
    x = jnp.ones((1, 5, 5, channels), dtype=jnp.float32)
    y_module = conv(x)

    # Build kernel_effective by injecting j_d in the constrained central entries
    kernel_effective = conv._set_jd_constraint(conv.kernel)
    # compute reference conv with lax via same grouped conv call:
    # (we can use lax.conv_general_dilated via conv_forward only if shapes align,
    # but here kernel_effective is grouped-layout; we can call the layer's own conv implementation:
    # emulate grouped conv using lax.conv_general_dilated directly)
    x_pad = pad_2d(x, kh // 2, kw // 2, "constant")

    y_ref = jax.lax.conv_general_dilated(
        lhs=x_pad,
        rhs=kernel_effective,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=groups,
    )
    # shapes should match
    assert y_module.shape == y_ref.shape
    assert jnp.allclose(y_module, y_ref)


def test_conv2drecurrentdiscrete_backward_zeroes_constrained_center():
    # ensure backward kills updates on constrained central diagonal positions
    key = jax.random.PRNGKey(3)
    conv = Conv2DRecurrentDiscrete(
        channels=6,
        kernel_size=(3, 3),
        groups=3,
        j_d=1.0,
        threshold=0.0,
        key=key,
        padding_mode="constant",
    )

    x = _rand(jax.random.PRNGKey(10), (2, 8, 8, 6))
    y = conv(x)
    y_hat = jnp.zeros_like(y)

    upd = conv.backward(x, y, y_hat, gate=None)
    # upd is a Module-like object with kernel-shaped update
    assert hasattr(upd, "kernel")
    # central element should have zeros in constrained positions (same mask logic)
    ch, cw = conv.central_element
    mask = conv._central_diag_mask()
    center_update = upd.kernel[ch, cw, :, :]
    # constrained entries must be zero (they are killed)
    assert jnp.allclose(center_update * mask, 0.0)


def test_conv2dtranspose_forward_shape_and_backward_update_shape():
    key = jax.random.PRNGKey(4)
    k1, k2 = jax.random.split(key)

    x = _rand(k1, (2, 5, 6, 2))  # (N,H,W,Cin)

    trans = Conv2DTranspose(
        in_channels=2,
        out_channels=3,
        kernel_size=(3, 3),
        threshold=0.0,
        strength=1.0,
        key=k2,
        stride=2,
        padding_mode="constant",
    )

    y = trans(x)
    y_ref = conv_transpose_forward(x, trans.kernel, stride=trans.stride, padding_mode="constant")
    assert y.shape == y_ref.shape
    assert jnp.allclose(y, y_ref)

    # backward should not crash and must return kernel-shaped update
    y_hat = jnp.zeros_like(y)
    upd = trans.backward(x, y, y_hat, gate=None)
    assert isinstance(upd, Conv2DTranspose)
    assert upd.kernel.shape == trans.kernel.shape

    # jittable forward
    y_jit = jax.jit(lambda m, inp: m(inp))(trans, x)
    assert jnp.allclose(y_jit, y_ref)

    # jittable backward
    upd_jit = jax.jit(lambda m, a, b, c: m.backward(a, b, c, None))(trans, x, y, y_hat)
    assert upd_jit.kernel.shape == trans.kernel.shape
