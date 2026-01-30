import pytest
import jax
import jax.numpy as jnp

from darnax.modules.conv.utils import (
    fetch_tuple_from_arg,
    pad_2d,
    conv_forward,
    conv_backward,
    conv_backward_with_threshold,
    conv_transpose_forward,
)


def _rand(key, shape, dtype=jnp.float32):
    return jax.random.normal(key, shape, dtype=dtype)


def test_fetch_tuple_from_arg_int():
    assert fetch_tuple_from_arg(3) == (3, 3)


def test_fetch_tuple_from_arg_tuple():
    assert fetch_tuple_from_arg((2, 5)) == (2, 5)


def test_pad_2d_none_is_identity():
    key = jax.random.PRNGKey(0)
    x = _rand(key, (2, 4, 5, 3))
    y = pad_2d(x, 2, 1, None)
    assert y is x or jnp.array_equal(y, x)
    assert y.shape == x.shape


def test_pad_2d_constant_padding_shape_and_values():
    x = jnp.ones((1, 2, 3, 1), dtype=jnp.float32)
    y = pad_2d(x, pad_h=1, pad_w=2, mode="constant")
    assert y.shape == (1, 2 + 2 * 1, 3 + 2 * 2, 1)
    # inner region matches original
    assert jnp.allclose(y[:, 1:3, 2:5, :], x)
    # padded border is 0 for constant default
    assert jnp.allclose(y[:, 0, :, :], 0.0)
    assert jnp.allclose(y[:, -1, :, :], 0.0)
    assert jnp.allclose(y[:, :, 0, :], 0.0)
    assert jnp.allclose(y[:, :, 1, :], 0.0)
    assert jnp.allclose(y[:, :, -1, :], 0.0)
    assert jnp.allclose(y[:, :, -2, :], 0.0)


@pytest.mark.parametrize("kh,kw", [(1, 1), (3, 3), (5, 3)])
def test_conv_forward_shape_stride1_same_hw_for_odd_kernels(kh, kw):
    # With symmetric pad_2d(kh//2, kw//2), odd kernels preserve H,W for stride=1.
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    x = _rand(k1, (2, 7, 9, 4))
    w = _rand(k2, (kh, kw, 4, 6))
    y = conv_forward(x, w, stride=(1, 1), padding_mode="constant")
    assert y.shape == (2, 7, 9, 6)


def test_conv_backward_shape_matches_kernel():
    key = jax.random.PRNGKey(2)
    kx, ky = jax.random.split(key)
    x = _rand(kx, (3, 8, 8, 5))
    # construct y as any compatible shape; use conv_forward to ensure shape consistency
    w = _rand(ky, (3, 3, 5, 7))
    y = conv_forward(x, w, stride=(1, 1), padding_mode="constant")
    dW = conv_backward(
        x=x,
        y=y,
        kernel_shape=(3, 3),
        strides=(1, 1),
        padding_mode="constant",
    )
    assert dW.shape == (3, 3, 5, 7)


def test_conv_backward_with_threshold_all_on_equals_no_threshold():
    key = jax.random.PRNGKey(3)
    kx, ky, kh = jax.random.split(key, 3)

    x = _rand(kx, (2, 6, 6, 3))
    w = _rand(ky, (3, 3, 3, 4))
    y = conv_forward(x, w, stride=(1, 1), padding_mode="constant")

    # y_hat arbitrary; choose zeros so y * y_hat = 0 always < large threshold
    y_hat = jnp.zeros_like(y)
    big_threshold = jnp.asarray(1e9, dtype=y.dtype)

    dW_thr = conv_backward_with_threshold(
        x=x,
        y=y,
        y_hat=y_hat,
        threshold=big_threshold,
        kernel_shape=(3, 3),
        strides=(1, 1),
        padding_mode="constant",
    )
    dW = conv_backward(
        x=x,
        y=y,
        kernel_shape=(3, 3),
        strides=(1, 1),
        padding_mode="constant",
    )
    assert jnp.allclose(dW_thr, dW, atol=1e-5, rtol=1e-5)


def test_conv_backward_with_threshold_all_off_is_zero():
    key = jax.random.PRNGKey(4)
    kx, ky, kh = jax.random.split(key, 3)

    x = _rand(kx, (2, 6, 6, 3))
    w = _rand(ky, (3, 3, 3, 4))
    y = conv_forward(x, w, stride=(1, 1), padding_mode="constant")

    # Choose y_hat = y so y*y_hat = y^2 >= 0.
    # If threshold is negative, condition y*y_hat < threshold is always false.
    y_hat = y
    neg_threshold = jnp.asarray(-1e-6, dtype=y.dtype)

    dW_thr = conv_backward_with_threshold(
        x=x,
        y=y,
        y_hat=y_hat,
        threshold=neg_threshold,
        kernel_shape=(3, 3),
        strides=(1, 1),
        padding_mode="constant",
    )
    assert jnp.allclose(dW_thr, 0.0)


def test_conv_transpose_forward_smoke_and_differentiable():
    key = jax.random.PRNGKey(5)
    kx, kw = jax.random.split(key)
    x = _rand(kx, (2, 5, 6, 3))
    w = _rand(kw, (3, 3, 3, 4))

    y = conv_transpose_forward(x, w, stride=2, padding_mode="constant")
    assert y.ndim == 4
    assert y.shape[0] == 2
    assert y.shape[-1] == 4

    # Differentiability smoke: gradients exist and are finite
    def loss_wrt_w(W):
        yy = conv_transpose_forward(x, W, stride=2, padding_mode="constant")
        return jnp.mean(yy**2)

    gW = jax.grad(loss_wrt_w)(w)
    assert gW.shape == w.shape
    assert jnp.all(jnp.isfinite(gW))


def test_jittability_all_ops():
    key = jax.random.PRNGKey(6)
    kx, kw, ky = jax.random.split(key, 3)

    x = _rand(kx, (2, 7, 9, 4))
    w = _rand(kw, (3, 3, 4, 5))
    y = conv_forward(x, w, stride=(1, 1), padding_mode="constant")
    y_hat = _rand(ky, y.shape)

    # JIT conv_forward and conv_transpose_forward: padding_mode is a Python object => static.
    jit_conv_fwd = jax.jit(conv_forward, static_argnames=("stride", "padding_mode"))
    jit_conv_t = jax.jit(conv_transpose_forward, static_argnames=("stride", "padding_mode"))

    y1 = jit_conv_fwd(x, w, stride=(1, 1), padding_mode="constant")
    y2 = jit_conv_t(x, w, stride=2, padding_mode="constant")
    assert y1.shape == y.shape
    assert y2.ndim == 4

    # JIT conv_backward / conv_backward_with_threshold: tuples + padding_mode should be static.
    jit_bwd = jax.jit(
        conv_backward,
        static_argnames=("kernel_shape", "strides", "padding_mode", "lhs_dilation", "rhs_dilation"),
    )
    jit_bwd_thr = jax.jit(
        conv_backward_with_threshold,
        static_argnames=("kernel_shape", "strides", "padding_mode", "lhs_dilation", "rhs_dilation"),
    )

    dW1 = jit_bwd(
        x=x,
        y=y,
        kernel_shape=(3, 3),
        strides=(1, 1),
        padding_mode="constant",
        lhs_dilation=(1, 1),
        rhs_dilation=(1, 1),
    )
    dW2 = jit_bwd_thr(
        x=x,
        y=y,
        y_hat=y_hat,
        threshold=jnp.asarray(1.0, dtype=x.dtype),
        kernel_shape=(3, 3),
        strides=(1, 1),
        padding_mode="constant",
        lhs_dilation=(1, 1),
        rhs_dilation=(1, 1),
    )
    assert dW1.shape == (3, 3, 4, 5)
    assert dW2.shape == (3, 3, 4, 5)
    assert jnp.all(jnp.isfinite(dW1))
    assert jnp.all(jnp.isfinite(dW2))
