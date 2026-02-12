"""
Tests for convolutional modules with Hebbian learning.

These tests verify:
1. Forward pass shapes and correctness
2. Backward pass (update) shapes and structure
3. JIT compilation compatibility
4. Diagonal constraint enforcement in recurrent layers
5. Variance normalization properties
"""

import jax
import jax.numpy as jnp
import pytest

from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete, Conv2DTranspose
from darnax.modules.conv.utils import (
    conv_forward,
    conv_transpose_forward,
)


def _rand(key, shape, dtype=jnp.float32):
    """Generate random normal array."""
    return jax.random.normal(key, shape, dtype=dtype)


# ============================================================================
# Conv2D Tests
# ============================================================================


def test_conv2d_forward_shape_and_jittable():
    """Test Conv2D forward pass produces correct shapes and is JIT-compatible."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    x = _rand(k1, (2, 8, 10, 3))

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

    # Forward via module should match direct conv_forward call
    y_module = conv(x)
    y_ref = conv.strength * conv_forward(x, conv.kernel, stride=(1, 1), padding_mode="constant")
    assert y_module.shape == y_ref.shape
    assert jnp.allclose(y_module, y_ref)

    # JIT compilation should work
    jit_call = jax.jit(lambda m, inp: m(inp))
    y_jit = jit_call(conv, x)
    assert jnp.allclose(y_jit, y_ref)


def test_conv2d_backward_returns_kernel_shaped_update():
    """Test Conv2D backward returns properly shaped update tree."""
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
        lr=0.01,
        weight_decay=0.001,
    )

    # Fake forward / supervision signals
    y = conv(x)
    y_hat = jnp.zeros_like(y)

    upd = conv.backward(x, y, y_hat, gate=None)

    # Backward should return same type with same pytree structure
    assert isinstance(upd, Conv2D)
    assert hasattr(upd, "kernel")
    assert upd.kernel.shape == conv.kernel.shape

    # All other leaves should be zero
    assert jnp.allclose(upd.threshold, 0.0)
    assert jnp.allclose(upd.lr, 0.0)
    assert jnp.allclose(upd.weight_decay, 0.0)


def test_conv2d_backward_jittable():
    """Test Conv2D backward is JIT-compatible."""
    key = jax.random.PRNGKey(2)
    kx, kw = jax.random.split(key)
    x = _rand(kx, (2, 7, 9, 4))

    conv = Conv2D(
        in_channels=4,
        out_channels=6,
        kernel_size=(3, 3),
        threshold=0.1,
        strength=0.5,
        key=kw,
        padding_mode="constant",
    )

    y = conv(x)
    y_hat = jnp.zeros_like(y)

    # JIT-compiled backward should work
    jit_bwd = jax.jit(lambda module, a, b, c: module.backward(a, b, c, None))
    upd_jit = jit_bwd(conv, x, y, y_hat)

    assert isinstance(upd_jit, Conv2D)
    assert upd_jit.kernel.shape == conv.kernel.shape


def test_conv2d_threshold_gating():
    """Test that threshold properly gates updates.

    Note: We use the actual forward output shape rather than manually creating
    y and y_hat, because the output shape depends on padding_mode:
    - padding_mode="constant": output shape ≈ input shape
    - padding_mode=None: output shape shrinks by (kernel_size - 1)
    """
    key = jax.random.PRNGKey(3)
    kx, kw = jax.random.split(key)

    # Create predictable input
    x = jnp.ones((1, 4, 4, 2))

    conv = Conv2D(
        in_channels=2,
        out_channels=2,
        kernel_size=3,
        threshold=0.0,  # Only negative products contribute
        strength=1.0,
        key=kw,
        lr=1.0,
        weight_decay=0.0,
        padding_mode="constant",  # Explicit padding mode
    )

    # Get actual output shape from forward pass
    y_actual = conv(x)

    # Create signals where product is always positive (no updates)
    y = jnp.ones_like(y_actual)
    y_hat = jnp.ones_like(y_actual)

    upd_no_gate = conv.backward(x, y, y_hat)

    # Now create signals where product is always negative (full updates)
    y_neg = jnp.ones_like(y_actual)
    y_hat_neg = -jnp.ones_like(y_actual)

    upd_full_gate = conv.backward(x, y_neg, y_hat_neg)

    # The gated version should have larger magnitude updates
    assert jnp.linalg.norm(upd_full_gate.kernel) > jnp.linalg.norm(upd_no_gate.kernel)


def test_conv2d_variance_normalization():
    """Test that gradient variance is approximately unit when inputs have unit variance."""
    key = jax.random.PRNGKey(4)
    kx, kw = jax.random.split(key)

    # Create unit variance inputs
    N, H, W = 32, 16, 16  # Large batch for stable variance estimate
    x = _rand(kx, (N, H, W, 8))
    y_hat = _rand(kx, (N, H, W, 16))

    x = jnp.sign((x - x.mean()) / x.std())
    y_hat = (y_hat - y_hat.mean()) / y_hat.std()
    y = jnp.sign(y_hat)

    conv = Conv2D(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        threshold=1e10,  # All updates pass
        strength=1.0,
        key=kw,
        lr=1.0,
        weight_decay=0.0,  # No weight decay for clean test
        padding_mode="constant",
    )

    upd = conv.backward(x, y, y_hat)

    # Gradient should have approximately unit variance
    grad_var = jnp.var(upd.kernel)
    assert 0.5 < grad_var < 2.0, f"Gradient variance {grad_var} outside expected range"


def test_conv2d_weight_decay_scaling():
    """Test that weight decay is scaled consistently with gradient."""
    key = jax.random.PRNGKey(5)
    kx, kw = jax.random.split(key)

    x = _rand(kx, (4, 8, 8, 4))

    conv = Conv2D(
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        threshold=1e10,
        strength=1.0,
        key=kw,
        lr=0.0,  # Turn off learning to isolate weight decay
        weight_decay=1.0,
        padding_mode="constant",
    )

    y = conv(x)
    y_hat = jnp.zeros_like(y)

    upd = conv.backward(x, y, y_hat)

    # Update should be proportional to kernel (weight decay only)
    # Normalized by 1/sqrt(N * Ho * Wo)
    N, Ho, Wo = y.shape[:3]
    expected_scale = 1.0 / jnp.sqrt(N * Ho * Wo)

    # Check that the update direction aligns with kernel
    kernel_norm = conv.kernel / jnp.linalg.norm(conv.kernel)
    upd_norm = upd.kernel / jnp.linalg.norm(upd.kernel)

    # Should be nearly parallel (cosine similarity close to 1)
    cosine_sim = jnp.sum(kernel_norm * upd_norm)
    assert cosine_sim > 0.99, f"Weight decay not aligned with kernel: {cosine_sim}"


# ============================================================================
# Conv2DRecurrentDiscrete Tests
# ============================================================================


def test_conv2drecurrentdiscrete_forward_shape():
    """Test recurrent conv forward produces correct shapes."""
    key = jax.random.PRNGKey(10)
    channels = 8
    groups = 2

    conv = Conv2DRecurrentDiscrete(
        channels=channels,
        kernel_size=3,
        groups=groups,
        j_d=1.0,
        threshold=0.0,
        key=key,
        padding_mode="constant",
        lr=0.01,
        weight_decay=0.001,
    )

    x = _rand(jax.random.PRNGKey(11), (2, 10, 10, channels))
    y = conv(x)

    # Output should have same shape as input (recurrent)
    assert y.shape == x.shape


def test_conv2drecurrentdiscrete_jd_constraint_initialization():
    """Test that j_d diagonal constraint is properly set at initialization."""
    key = jax.random.PRNGKey(20)
    channels = 6
    groups = 3
    j_d = 2.5

    conv = Conv2DRecurrentDiscrete(
        channels=channels,
        kernel_size=3,
        groups=groups,
        j_d=j_d,
        threshold=0.0,
        key=key,
        padding_mode="constant",
        lr=0.01,
        weight_decay=0.001,
    )

    # Check that central element has j_d on the diagonal
    ch, cw = conv.central_element
    center = conv.kernel[ch, cw, :, :]

    # Get diagonal mask
    mask = conv._central_diag_mask()

    # Extract diagonal values
    diagonal_vals = center * mask

    # All diagonal entries should be j_d
    num_diag_entries = int(mask.sum())
    expected_sum = j_d * num_diag_entries
    actual_sum = diagonal_vals.sum()

    assert jnp.allclose(actual_sum, expected_sum), (
        f"Diagonal sum {actual_sum} != expected {expected_sum}"
    )


def test_conv2drecurrentdiscrete_update_mask():
    """Test that update mask properly blocks constrained parameters."""
    key = jax.random.PRNGKey(30)
    channels = 4
    groups = 2

    conv = Conv2DRecurrentDiscrete(
        channels=channels,
        kernel_size=3,
        groups=groups,
        j_d=1.0,
        threshold=0.0,
        key=key,
        padding_mode="constant",
        lr=0.01,
        weight_decay=0.001,
    )

    ch, cw = conv.central_element
    mask = conv._central_diag_mask()
    update_mask = conv.update_mask

    # At central element, diagonal positions should have 0 in update_mask
    center_update_mask = update_mask[ch, cw, :, :]

    # Where diagonal mask is 1, update mask should be 0
    assert jnp.allclose(center_update_mask * mask, 0.0)

    # Elsewhere, update mask should be 1
    non_center_mask = jnp.ones_like(update_mask)
    non_center_mask = non_center_mask.at[ch, cw, :, :].set(0.0)
    assert jnp.allclose(update_mask * non_center_mask, non_center_mask)


def test_conv2drecurrentdiscrete_backward_preserves_jd():
    """Test that backward pass does not update constrained diagonal."""
    key = jax.random.PRNGKey(40)
    channels = 6
    groups = 3
    j_d = 1.5

    conv = Conv2DRecurrentDiscrete(
        channels=channels,
        kernel_size=3,
        groups=groups,
        j_d=j_d,
        threshold=0.0,
        key=key,
        padding_mode="constant",
        lr=0.01,
        weight_decay=0.001,
    )

    x = _rand(jax.random.PRNGKey(41), (2, 8, 8, channels))
    y = conv(x)
    y_hat = _rand(jax.random.PRNGKey(42), y.shape)

    upd = conv.backward(x, y, y_hat, gate=None)

    # Update at constrained positions should be zero
    ch, cw = conv.central_element
    mask = conv._central_diag_mask()
    center_update = upd.kernel[ch, cw, :, :]

    # Constrained entries must have zero update
    assert jnp.allclose(center_update * mask, 0.0)


def test_conv2drecurrentdiscrete_backward_jittable():
    """Test that recurrent conv backward is JIT-compatible."""
    key = jax.random.PRNGKey(50)

    conv = Conv2DRecurrentDiscrete(
        channels=8,
        kernel_size=3,
        groups=2,
        j_d=1.0,
        threshold=0.0,
        key=key,
        padding_mode="constant",
        lr=0.01,
        weight_decay=0.001,
    )

    x = _rand(jax.random.PRNGKey(51), (2, 6, 6, 8))
    y = conv(x)
    y_hat = jnp.zeros_like(y)

    # JIT-compiled backward
    jit_bwd = jax.jit(lambda m, a, b, c: m.backward(a, b, c, None))
    upd_jit = jit_bwd(conv, x, y, y_hat)

    assert isinstance(upd_jit, Conv2DRecurrentDiscrete)
    assert upd_jit.kernel.shape == conv.kernel.shape


def test_conv2drecurrentdiscrete_diag_group_blocks():
    """Test that diagonal group block extraction works correctly."""
    key = jax.random.PRNGKey(60)
    channels = 8
    groups = 4

    conv = Conv2DRecurrentDiscrete(
        channels=channels,
        kernel_size=3,
        groups=groups,
        j_d=1.0,
        threshold=0.0,
        key=key,
        padding_mode="constant",
        lr=0.01,
        weight_decay=0.001,
    )

    # Create a dummy full gradient
    kh, kw = conv.kernel_size
    dW_full = _rand(jax.random.PRNGKey(61), (kh, kw, channels, channels))

    # Extract diagonal blocks
    dW_diag = conv._diag_group_blocks(dW_full)

    # Should have shape (kh, kw, cin_g, cout)
    cin_g = channels // groups
    assert dW_diag.shape == (kh, kw, cin_g, channels)


# ============================================================================
# Conv2DTranspose Tests
# ============================================================================


def test_conv2dtranspose_forward_upsampling():
    """Test that transposed conv properly upsamples."""
    key = jax.random.PRNGKey(70)
    k1, k2 = jax.random.split(key)

    x = _rand(k1, (2, 8, 8, 4))

    trans = Conv2DTranspose(
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        threshold=0.0,
        strength=1.0,
        key=k2,
        stride=2,  # 2x upsampling
        padding_mode="constant",
    )

    y = trans(x)

    # Output should be approximately 2x larger in spatial dimensions
    # (exact size depends on padding/kernel details)
    assert y.shape[1] > x.shape[1]  # Height increased
    assert y.shape[2] > x.shape[2]  # Width increased
    assert y.shape[3] == 8  # Correct output channels


def test_conv2dtranspose_forward_matches_utils():
    """Test that Conv2DTranspose forward matches conv_transpose_forward."""
    key = jax.random.PRNGKey(80)
    k1, k2 = jax.random.split(key)

    x = _rand(k1, (2, 5, 6, 2))

    trans = Conv2DTranspose(
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        threshold=0.0,
        strength=1.0,
        key=k2,
        stride=2,
        padding_mode="constant",
    )

    y_module = trans(x)
    y_ref = conv_transpose_forward(x, trans.kernel, stride=trans.stride, padding_mode="constant")

    assert y_module.shape == y_ref.shape
    assert jnp.allclose(y_module, y_ref)


def test_conv2dtranspose_backward_shape():
    """Test that transposed conv backward returns correct shape."""
    key = jax.random.PRNGKey(90)
    k1, k2 = jax.random.split(key)

    x = _rand(k1, (2, 4, 4, 3))

    trans = Conv2DTranspose(
        in_channels=3,
        out_channels=5,
        kernel_size=3,
        threshold=0.0,
        strength=1.0,
        key=k2,
        stride=2,
        padding_mode="constant",
    )

    y = trans(x)
    y_hat = jnp.zeros_like(y)

    upd = trans.backward(x, y, y_hat, gate=None)

    assert isinstance(upd, Conv2DTranspose)
    assert upd.kernel.shape == trans.kernel.shape


def test_conv2dtranspose_jittable():
    """Test that Conv2DTranspose is fully JIT-compatible."""
    key = jax.random.PRNGKey(100)
    k1, k2 = jax.random.split(key)

    x = _rand(k1, (2, 4, 4, 2))

    trans = Conv2DTranspose(
        in_channels=2,
        out_channels=4,
        kernel_size=3,
        threshold=0.0,
        strength=1.0,
        key=k2,
        stride=2,
        padding_mode="constant",
    )

    # JIT forward
    y_jit = jax.jit(lambda m, inp: m(inp))(trans, x)
    y_ref = trans(x)
    assert jnp.allclose(y_jit, y_ref)

    # JIT backward
    y_hat = jnp.zeros_like(y_jit)
    upd_jit = jax.jit(lambda m, a, b, c: m.backward(a, b, c, None))(trans, x, y_jit, y_hat)
    assert upd_jit.kernel.shape == trans.kernel.shape


# ============================================================================
# Integration Tests
# ============================================================================


def test_all_modules_pytree_compatible():
    """Test that all modules are valid pytrees for JAX transformations."""
    key = jax.random.PRNGKey(200)
    k1, k2, k3 = jax.random.split(key, 3)

    modules = [
        Conv2D(3, 5, 3, 0.0, 1.0, k1, padding_mode="constant"),
        Conv2DRecurrentDiscrete(8, 3, 2, 1.0, 0.0, k2, lr=0.01, weight_decay=0.001),
        Conv2DTranspose(4, 6, 3, 0.0, 1.0, k3, stride=2),
    ]

    for module in modules:
        # Should be able to flatten/unflatten
        leaves, treedef = jax.tree_util.tree_flatten(module)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        # Test with dummy input
        if isinstance(module, Conv2DRecurrentDiscrete):
            x = _rand(jax.random.PRNGKey(201), (1, 4, 4, module.channels))
        else:
            x = _rand(jax.random.PRNGKey(201), (1, 4, 4, module.in_channels))

        y1 = module(x)
        y2 = reconstructed(x)
        assert jnp.allclose(y1, y2)


def test_conv2d_different_strides():
    """Test Conv2D with various stride configurations."""
    key = jax.random.PRNGKey(300)
    k1, k2 = jax.random.split(key)

    x = _rand(k1, (1, 16, 16, 3))

    for stride in [1, 2, (1, 2), (2, 1)]:
        conv = Conv2D(3, 5, 3, 0.0, 1.0, k2, stride=stride, padding_mode="constant")
        y = conv(x)

        # Should not crash
        assert y.ndim == 4
        assert y.shape[0] == 1  # Batch preserved
        assert y.shape[3] == 5  # Output channels correct


def test_conv2d_different_kernel_sizes():
    """Test Conv2D with various kernel sizes."""
    key = jax.random.PRNGKey(400)
    k1, k2 = jax.random.split(key)

    x = _rand(k1, (1, 8, 8, 2))

    for kernel_size in [1, 3, 5, (3, 5), (5, 3)]:
        conv = Conv2D(2, 4, kernel_size, 0.0, 1.0, k2, padding_mode="constant")
        y = conv(x)

        # Should not crash
        assert y.ndim == 4
        assert y.shape[3] == 4  # Output channels correct
