"""Tests for convolutional utility functions.

Covers:
- Helper functions (fetch_tuple_from_arg, pad_2d)
- Forward passes (conv_forward, conv_transpose_forward)
- Backward accumulators (conv_backward, conv_backward_with_threshold,
  conv_transpose_backward_with_threshold)
- Block-diagonal grouped expansion (_expand_grouped_to_full)
- Variance normalization properties
- Threshold gating correctness
- JIT compatibility
"""

import jax
import jax.numpy as jnp
import pytest

from darnax.modules.conv.utils import (
    fetch_tuple_from_arg,
    pad_2d,
    conv_forward,
    conv_transpose_forward,
    conv_backward,
    conv_backward_with_threshold,
    conv_transpose_backward_with_threshold,
    _expand_grouped_to_full,
)


def _rand(key, shape, dtype=jnp.float32):
    return jax.random.normal(key, shape, dtype=dtype)


# ============================================================================
# fetch_tuple_from_arg
# ============================================================================


def test_fetch_tuple_from_int():
    assert fetch_tuple_from_arg(3) == (3, 3)


def test_fetch_tuple_from_tuple():
    assert fetch_tuple_from_arg((1, 2)) == (1, 2)


def test_fetch_tuple_from_list():
    assert fetch_tuple_from_arg([4, 5]) == (4, 5)


def test_fetch_tuple_rejects_bad_input():
    with pytest.raises(ValueError):
        fetch_tuple_from_arg((1, 2, 3))
    with pytest.raises(ValueError):
        fetch_tuple_from_arg("ab")


# ============================================================================
# pad_2d
# ============================================================================


def test_pad_2d_constant():
    x = jnp.ones((1, 4, 4, 1))
    out = pad_2d(x, 1, 2, "constant")
    assert out.shape == (1, 6, 8, 1)
    # corners should be zero (constant pad)
    assert out[0, 0, 0, 0] == 0.0
    # interior should be one
    assert out[0, 1, 2, 0] == 1.0


def test_pad_2d_none_is_identity():
    x = jnp.ones((2, 3, 3, 2))
    out = pad_2d(x, 5, 5, None)
    assert out is x


def test_pad_2d_reflect():
    x = jnp.arange(4.0).reshape(1, 1, 4, 1)
    out = pad_2d(x, 0, 1, "reflect")
    # reflect padding of [0,1,2,3] with width 1 -> [1,0,1,2,3,2]
    assert out.shape == (1, 1, 6, 1)
    assert float(out[0, 0, 0, 0]) == 1.0
    assert float(out[0, 0, -1, 0]) == 2.0


# ============================================================================
# conv_forward
# ============================================================================


def test_conv_forward_shape_no_padding():
    x = _rand(jax.random.PRNGKey(0), (2, 8, 8, 3))
    k = _rand(jax.random.PRNGKey(1), (3, 3, 3, 16))
    y = conv_forward(x, k, stride=1, padding_mode=None)
    # No padding: output shrinks by kernel_size - 1
    assert y.shape == (2, 6, 6, 16)


def test_conv_forward_shape_with_padding():
    x = _rand(jax.random.PRNGKey(0), (2, 8, 8, 3))
    k = _rand(jax.random.PRNGKey(1), (3, 3, 3, 16))
    y = conv_forward(x, k, stride=1, padding_mode="constant")
    # Same-size output with kh//2 padding
    assert y.shape == (2, 8, 8, 16)


def test_conv_forward_stride2():
    x = _rand(jax.random.PRNGKey(0), (1, 16, 16, 4))
    k = _rand(jax.random.PRNGKey(1), (3, 3, 4, 8))
    y = conv_forward(x, k, stride=2, padding_mode="constant")
    assert y.shape == (1, 8, 8, 8)


def test_conv_forward_asymmetric_kernel():
    x = _rand(jax.random.PRNGKey(0), (1, 10, 10, 2))
    k = _rand(jax.random.PRNGKey(1), (3, 5, 2, 4))
    y = conv_forward(x, k, stride=1, padding_mode="constant")
    assert y.shape == (1, 10, 10, 4)


def test_conv_forward_identity_kernel():
    """A 1x1 identity-like kernel should pass through input channels."""
    x = _rand(jax.random.PRNGKey(0), (1, 4, 4, 3))
    k = jnp.eye(3).reshape(1, 1, 3, 3)
    y = conv_forward(x, k, stride=1, padding_mode="constant")
    assert jnp.allclose(y, x)


# ============================================================================
# conv_transpose_forward
# ============================================================================


def test_conv_transpose_forward_upsamples():
    x = _rand(jax.random.PRNGKey(0), (1, 8, 8, 4))
    k = _rand(jax.random.PRNGKey(1), (3, 3, 4, 6))
    y = conv_transpose_forward(x, k, stride=2, padding_mode="constant")
    # stride=2 should roughly double spatial dims
    assert y.shape[0] == 1
    assert y.shape[1] > 8
    assert y.shape[2] > 8
    assert y.shape[3] == 6


def test_conv_transpose_forward_stride1_with_padding():
    x = _rand(jax.random.PRNGKey(0), (2, 6, 6, 3))
    k = _rand(jax.random.PRNGKey(1), (3, 3, 3, 5))
    y = conv_transpose_forward(x, k, stride=1, padding_mode="constant")
    # stride=1 transposed conv with padding should produce an output
    assert y.ndim == 4
    assert y.shape[0] == 2
    assert y.shape[3] == 5


def test_conv_transpose_forward_channel_mismatch():
    x = _rand(jax.random.PRNGKey(0), (1, 4, 4, 3))
    k = _rand(jax.random.PRNGKey(1), (3, 3, 5, 2))  # cin=5 != 3
    with pytest.raises(AssertionError):
        conv_transpose_forward(x, k, stride=1)


# ============================================================================
# _expand_grouped_to_full
# ============================================================================


def test_expand_grouped_block_diagonal_structure():
    """Verify that cross-group entries are zero and within-group entries are preserved."""
    groups, Cin_per, Cout_per = 3, 2, 4
    kh, kw = 3, 3
    key = jax.random.PRNGKey(0)
    accum_g = _rand(key, (kh, kw, groups, Cin_per, Cout_per))

    full = _expand_grouped_to_full(accum_g, groups, Cin_per, Cout_per, kh, kw)
    assert full.shape == (kh, kw, groups * Cin_per, groups * Cout_per)

    # Check each group's block is placed correctly and cross-group is zero
    for g in range(groups):
        cin_s, cin_e = g * Cin_per, (g + 1) * Cin_per
        cout_s, cout_e = g * Cout_per, (g + 1) * Cout_per

        # Within-group block should match
        expected_block = accum_g[:, :, g, :, :]
        actual_block = full[:, :, cin_s:cin_e, cout_s:cout_e]
        assert jnp.allclose(actual_block, expected_block), f"Group {g} block mismatch"

    # Cross-group entries should be zero
    for g_in in range(groups):
        for g_out in range(groups):
            if g_in == g_out:
                continue
            cin_s = g_in * Cin_per
            cin_e = (g_in + 1) * Cin_per
            cout_s = g_out * Cout_per
            cout_e = (g_out + 1) * Cout_per
            cross = full[:, :, cin_s:cin_e, cout_s:cout_e]
            assert jnp.allclose(cross, 0.0), f"Cross-group ({g_in},{g_out}) not zero"


def test_expand_grouped_single_group_is_identity():
    """With groups=1, expansion should be a simple reshape."""
    kh, kw, Cin, Cout = 3, 3, 4, 8
    key = jax.random.PRNGKey(1)
    accum_g = _rand(key, (kh, kw, 1, Cin, Cout))

    full = _expand_grouped_to_full(accum_g, 1, Cin, Cout, kh, kw)
    assert full.shape == (kh, kw, Cin, Cout)
    assert jnp.allclose(full, accum_g.reshape(kh, kw, Cin, Cout))


def test_expand_grouped_jittable():
    groups, Cin_per, Cout_per = 4, 3, 5
    kh, kw = 3, 3
    accum_g = _rand(jax.random.PRNGKey(2), (kh, kw, groups, Cin_per, Cout_per))

    f = jax.jit(lambda a: _expand_grouped_to_full(a, groups, Cin_per, Cout_per, kh, kw))
    full_jit = f(accum_g)
    full_ref = _expand_grouped_to_full(accum_g, groups, Cin_per, Cout_per, kh, kw)
    assert jnp.allclose(full_jit, full_ref)


# ============================================================================
# conv_backward
# ============================================================================


def test_conv_backward_shape_ungrouped():
    x = _rand(jax.random.PRNGKey(0), (4, 8, 8, 6))
    y = _rand(jax.random.PRNGKey(1), (4, 8, 8, 10))
    dW = conv_backward(x, y, kernel_shape=(3, 3), padding_mode="constant")
    assert dW.shape == (3, 3, 6, 10)


def test_conv_backward_shape_grouped():
    x = _rand(jax.random.PRNGKey(0), (4, 8, 8, 12))
    y = _rand(jax.random.PRNGKey(1), (4, 8, 8, 12))
    dW = conv_backward(x, y, kernel_shape=(3, 3), groups=3, padding_mode="constant")
    assert dW.shape == (3, 3, 12, 12)


def test_conv_backward_grouped_block_diagonal():
    """Grouped backward should produce block-diagonal gradient."""
    groups = 4
    Cin, Cout = 8, 8
    Cin_per = Cin // groups
    Cout_per = Cout // groups

    x = _rand(jax.random.PRNGKey(0), (2, 6, 6, Cin))
    y = _rand(jax.random.PRNGKey(1), (2, 6, 6, Cout))
    dW = conv_backward(x, y, kernel_shape=(3, 3), groups=groups, padding_mode="constant")

    for g_in in range(groups):
        for g_out in range(groups):
            if g_in == g_out:
                continue
            block = dW[
                :,
                :,
                g_in * Cin_per : (g_in + 1) * Cin_per,
                g_out * Cout_per : (g_out + 1) * Cout_per,
            ]
            assert jnp.allclose(block, 0.0), f"Cross-group ({g_in},{g_out}) not zero"


def test_conv_backward_variance_normalization():
    """Gradient variance ≈ 1 when x and y have unit variance."""
    N, H, W, Cin, Cout = 64, 16, 16, 8, 16
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    x = _rand(k1, (N, H, W, Cin))
    y = _rand(k2, (N, H, W, Cout))

    dW = conv_backward(x, y, kernel_shape=(3, 3), padding_mode="constant")

    var = float(jnp.var(dW))
    assert 0.5 < var < 2.0, f"Gradient variance {var} outside [0.5, 2.0]"


def test_conv_backward_linearity_in_y():
    """Backward should be linear in y: backward(x, a*y) = a * backward(x, y)."""
    x = _rand(jax.random.PRNGKey(0), (2, 8, 8, 4))
    y = _rand(jax.random.PRNGKey(1), (2, 8, 8, 6))
    a = 3.7

    dW1 = conv_backward(x, y, kernel_shape=(3, 3), padding_mode="constant")
    dW2 = conv_backward(x, a * y, kernel_shape=(3, 3), padding_mode="constant")

    assert jnp.allclose(dW2, a * dW1, atol=1e-5)


def test_conv_backward_jittable():
    x = _rand(jax.random.PRNGKey(0), (2, 8, 8, 4))
    y = _rand(jax.random.PRNGKey(1), (2, 8, 8, 6))

    f = jax.jit(
        lambda a, b: conv_backward(a, b, kernel_shape=(3, 3), groups=1, padding_mode="constant")
    )
    dW_jit = f(x, y)
    dW_ref = conv_backward(x, y, kernel_shape=(3, 3), padding_mode="constant")
    assert jnp.allclose(dW_jit, dW_ref)


def test_conv_backward_grouped_jittable():
    x = _rand(jax.random.PRNGKey(0), (2, 8, 8, 8))
    y = _rand(jax.random.PRNGKey(1), (2, 8, 8, 8))

    f = jax.jit(
        lambda a, b: conv_backward(a, b, kernel_shape=(3, 3), groups=4, padding_mode="constant")
    )
    dW_jit = f(x, y)
    dW_ref = conv_backward(x, y, kernel_shape=(3, 3), groups=4, padding_mode="constant")
    assert jnp.allclose(dW_jit, dW_ref)


# ============================================================================
# conv_backward_with_threshold
# ============================================================================


def test_threshold_all_pass_matches_conv_backward():
    """With threshold=+inf, all positions pass and result should match conv_backward."""
    x = _rand(jax.random.PRNGKey(0), (4, 8, 8, 6))
    y = _rand(jax.random.PRNGKey(1), (4, 8, 8, 10))
    y_hat = _rand(jax.random.PRNGKey(2), (4, 8, 8, 10))

    dW_thresh = conv_backward_with_threshold(
        x,
        y,
        y_hat,
        threshold=jnp.float32(1e10),
        kernel_shape=(3, 3),
        padding_mode="constant",
    )
    dW_ref = conv_backward(x, y, kernel_shape=(3, 3), padding_mode="constant")

    assert jnp.allclose(dW_thresh, dW_ref, atol=1e-5)


def test_threshold_none_pass():
    """With threshold=-inf, no positions pass and gradient should be zero."""
    x = _rand(jax.random.PRNGKey(0), (2, 6, 6, 4))
    y = _rand(jax.random.PRNGKey(1), (2, 6, 6, 8))
    y_hat = _rand(jax.random.PRNGKey(2), (2, 6, 6, 8))

    dW = conv_backward_with_threshold(
        x,
        y,
        y_hat,
        threshold=jnp.float32(-1e10),
        kernel_shape=(3, 3),
        padding_mode="constant",
    )
    assert jnp.allclose(dW, 0.0)


def test_threshold_gating_mask_correctness():
    """Verify gating: only positions where y*y_hat < threshold contribute."""
    N, H, W, Cin, Cout = 1, 4, 4, 2, 2

    x = jnp.ones((N, H, W, Cin))

    # y positive everywhere
    y = jnp.ones((N, H, W, Cout))
    # y_hat: first channel negative, second channel positive
    y_hat = jnp.concatenate(
        [
            -jnp.ones((N, H, W, 1)),
            jnp.ones((N, H, W, 1)),
        ],
        axis=-1,
    )

    # y*y_hat: first channel = -1 (< 0), second channel = +1 (>= 0)
    # threshold=0 -> only first output channel passes
    dW = conv_backward_with_threshold(
        x,
        y,
        y_hat,
        threshold=jnp.float32(0.0),
        kernel_shape=(3, 3),
        padding_mode="constant",
    )

    # Second output channel column should be zero
    assert jnp.allclose(dW[:, :, :, 1], 0.0)
    # First output channel column should be nonzero
    assert not jnp.allclose(dW[:, :, :, 0], 0.0)


def test_threshold_grouped():
    """Threshold backward with groups should produce block-diagonal output."""
    groups = 2
    Cin, Cout = 4, 4
    Cin_per = Cin // groups
    Cout_per = Cout // groups

    x = _rand(jax.random.PRNGKey(0), (2, 6, 6, Cin))
    y = _rand(jax.random.PRNGKey(1), (2, 6, 6, Cout))
    y_hat = _rand(jax.random.PRNGKey(2), (2, 6, 6, Cout))

    dW = conv_backward_with_threshold(
        x,
        y,
        y_hat,
        threshold=jnp.float32(1e10),
        kernel_shape=(3, 3),
        groups=groups,
        padding_mode="constant",
    )
    assert dW.shape == (3, 3, Cin, Cout)

    # Cross-group blocks must be zero
    for g_in in range(groups):
        for g_out in range(groups):
            if g_in == g_out:
                continue
            block = dW[
                :,
                :,
                g_in * Cin_per : (g_in + 1) * Cin_per,
                g_out * Cout_per : (g_out + 1) * Cout_per,
            ]
            assert jnp.allclose(block, 0.0)


def test_threshold_variance_normalization():
    """With all positions passing, variance ≈ 1 for unit-variance inputs."""
    N, H, W, Cin, Cout = 64, 16, 16, 8, 16
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)

    x = _rand(k1, (N, H, W, Cin))
    y = _rand(k2, (N, H, W, Cout))
    y_hat = _rand(k3, (N, H, W, Cout))

    dW = conv_backward_with_threshold(
        x,
        y,
        y_hat,
        threshold=jnp.float32(1e10),
        kernel_shape=(3, 3),
        padding_mode="constant",
    )
    var = float(jnp.var(dW))
    assert 0.5 < var < 2.0, f"Gradient variance {var} outside [0.5, 2.0]"


def test_threshold_jittable():
    x = _rand(jax.random.PRNGKey(0), (2, 6, 6, 4))
    y = _rand(jax.random.PRNGKey(1), (2, 6, 6, 4))
    y_hat = _rand(jax.random.PRNGKey(2), (2, 6, 6, 4))

    f = jax.jit(
        lambda a, b, c: conv_backward_with_threshold(
            a,
            b,
            c,
            threshold=jnp.float32(0.0),
            kernel_shape=(3, 3),
            groups=2,
            padding_mode="constant",
        )
    )
    dW_jit = f(x, y, y_hat)
    dW_ref = conv_backward_with_threshold(
        x,
        y,
        y_hat,
        threshold=jnp.float32(0.0),
        kernel_shape=(3, 3),
        groups=2,
        padding_mode="constant",
    )
    assert jnp.allclose(dW_jit, dW_ref)


# ============================================================================
# conv_transpose_backward_with_threshold
# ============================================================================


def test_transpose_backward_shape():
    x = _rand(jax.random.PRNGKey(0), (2, 8, 8, 4))
    # Compute forward to get correct output shape
    k = _rand(jax.random.PRNGKey(1), (3, 3, 4, 6))
    y_fwd = conv_transpose_forward(x, k, stride=2, padding_mode="constant")

    y = _rand(jax.random.PRNGKey(2), y_fwd.shape)
    y_hat = _rand(jax.random.PRNGKey(3), y_fwd.shape)

    dW = conv_transpose_backward_with_threshold(
        x,
        y,
        y_hat,
        threshold=jnp.float32(0.0),
        kernel_shape=(3, 3),
        stride=2,
        padding_mode="constant",
    )
    assert dW.shape == (3, 3, 4, 6)


def test_transpose_backward_none_pass():
    x = _rand(jax.random.PRNGKey(0), (2, 4, 4, 3))
    k = _rand(jax.random.PRNGKey(1), (3, 3, 3, 5))
    y_fwd = conv_transpose_forward(x, k, stride=2, padding_mode="constant")

    y = _rand(jax.random.PRNGKey(2), y_fwd.shape)
    y_hat = _rand(jax.random.PRNGKey(3), y_fwd.shape)

    dW = conv_transpose_backward_with_threshold(
        x,
        y,
        y_hat,
        threshold=jnp.float32(-1e10),
        kernel_shape=(3, 3),
        stride=2,
        padding_mode="constant",
    )
    assert jnp.allclose(dW, 0.0)


def test_transpose_backward_grouped_block_diagonal():
    groups = 2
    Cin, Cout = 4, 4
    Cin_per = Cin // groups
    Cout_per = Cout // groups

    x = _rand(jax.random.PRNGKey(0), (2, 4, 4, Cin))
    k = _rand(jax.random.PRNGKey(1), (3, 3, Cin, Cout))
    y_fwd = conv_transpose_forward(x, k, stride=2, padding_mode="constant")

    y = _rand(jax.random.PRNGKey(2), y_fwd.shape)
    y_hat = _rand(jax.random.PRNGKey(3), y_fwd.shape)

    dW = conv_transpose_backward_with_threshold(
        x,
        y,
        y_hat,
        threshold=jnp.float32(1e10),
        kernel_shape=(3, 3),
        stride=2,
        groups=groups,
        padding_mode="constant",
    )
    assert dW.shape == (3, 3, Cin, Cout)

    for g_in in range(groups):
        for g_out in range(groups):
            if g_in == g_out:
                continue
            block = dW[
                :,
                :,
                g_in * Cin_per : (g_in + 1) * Cin_per,
                g_out * Cout_per : (g_out + 1) * Cout_per,
            ]
            assert jnp.allclose(block, 0.0)


def test_transpose_backward_jittable():
    x = _rand(jax.random.PRNGKey(0), (2, 4, 4, 4))
    k = _rand(jax.random.PRNGKey(1), (3, 3, 4, 4))
    y_fwd = conv_transpose_forward(x, k, stride=2, padding_mode="constant")

    y = _rand(jax.random.PRNGKey(2), y_fwd.shape)
    y_hat = _rand(jax.random.PRNGKey(3), y_fwd.shape)

    f = jax.jit(
        lambda a, b, c: conv_transpose_backward_with_threshold(
            a,
            b,
            c,
            threshold=jnp.float32(0.0),
            kernel_shape=(3, 3),
            stride=2,
            groups=2,
            padding_mode="constant",
        )
    )
    dW_jit = f(x, y, y_hat)
    dW_ref = conv_transpose_backward_with_threshold(
        x,
        y,
        y_hat,
        threshold=jnp.float32(0.0),
        kernel_shape=(3, 3),
        stride=2,
        groups=2,
        padding_mode="constant",
    )
    assert jnp.allclose(dW_jit, dW_ref)


# ============================================================================
# Cross-function consistency
# ============================================================================


def test_conv_backward_matches_manual_einsum():
    """Verify ungrouped conv_backward against a direct einsum on patches."""
    N, H, W, Cin, Cout = 2, 6, 6, 3, 5
    kh, kw = 3, 3
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))

    x = _rand(k1, (N, H, W, Cin))
    y = _rand(k2, (N, H, W, Cout))

    dW = conv_backward(x, y, kernel_shape=(kh, kw), padding_mode="constant")

    # Manual: pad, extract patches, einsum, scale
    x_pad = pad_2d(x, kh // 2, kw // 2, "constant")

    patches = jax.lax.conv_general_dilated_patches(
        x_pad,
        filter_shape=(kh, kw),
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    # patches: (N, H, W, Cin*kh*kw) packed as (Cin, kh, kw)
    patches = patches.reshape(N, H, W, Cin, kh, kw)
    patches = jnp.transpose(patches, (0, 1, 2, 4, 5, 3))  # (N,H,W,kh,kw,Cin)

    accum = jnp.einsum("nhwklc, nhwo -> klco", patches, y)
    scale = 1.0 / jnp.sqrt(jnp.asarray(N * H * W, dtype=accum.dtype))
    expected = accum * scale

    assert jnp.allclose(dW, expected, atol=1e-6)


def test_grouped_backward_within_group_matches_ungrouped():
    """Each group's block in a grouped backward should match an ungrouped
    backward on that group's channel slice."""
    groups = 2
    Cin, Cout = 4, 6
    Cin_per = Cin // groups
    Cout_per = Cout // groups
    N, H, W = 2, 6, 6

    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    x = _rand(k1, (N, H, W, Cin))
    y = _rand(k2, (N, H, W, Cout))

    dW_grouped = conv_backward(
        x,
        y,
        kernel_shape=(3, 3),
        groups=groups,
        padding_mode="constant",
    )

    for g in range(groups):
        x_g = x[:, :, :, g * Cin_per : (g + 1) * Cin_per]
        y_g = y[:, :, :, g * Cout_per : (g + 1) * Cout_per]

        dW_g = conv_backward(
            x_g,
            y_g,
            kernel_shape=(3, 3),
            groups=1,
            padding_mode="constant",
        )

        actual_block = dW_grouped[
            :,
            :,
            g * Cin_per : (g + 1) * Cin_per,
            g * Cout_per : (g + 1) * Cout_per,
        ]
        assert jnp.allclose(actual_block, dW_g, atol=1e-5), f"Group {g} block mismatch"
