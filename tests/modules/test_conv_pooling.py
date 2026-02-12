import pytest
import jax
import jax.numpy as jnp
from darnax.modules.conv.pooling import (
    MajorityPooling,
    ConstantUnpooling,
    GlobalMajorityPooling,
    GlobalUnpooling,
)


# MajorityPooling tests


def test_basic_pooling_3x3_stride1():
    """Test that 3x3 pooling with stride 1 and padding preserves shape."""
    key = jax.random.PRNGKey(0)
    x = jnp.array(
        [
            [
                [[1], [-1], [1]],
                [[-1], [1], [-1]],
                [[1], [1], [1]],
            ]
        ],
        dtype=jnp.float32,
    )

    pooling = MajorityPooling(kernel_size=3, strength=1.0, key=key, stride=1, padding_mode="edge")
    out = pooling(x)

    assert out.shape == (1, 3, 3, 1)


def test_pooling_compression_stride_equals_kernel():
    """Test that pooling with stride=kernel_size compresses the image."""
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 6, 6, 1), dtype=jnp.float32)

    pooling = MajorityPooling(kernel_size=3, strength=1.0, key=key, stride=3, padding_mode="edge")
    out = pooling(x)

    assert out.shape == (1, 2, 2, 1)


def test_majority_vote_positive():
    """Test that majority of positive values gives +1."""
    key = jax.random.PRNGKey(0)
    x = jnp.array(
        [
            [
                [[1], [1], [1]],
                [[1], [-1], [-1]],
                [[1], [-1], [-1]],
            ]
        ],
        dtype=jnp.float32,
    )

    pooling = MajorityPooling(kernel_size=3, strength=1.0, key=key, stride=3, padding_mode="edge")
    out = pooling(x)

    assert out.shape == (1, 1, 1, 1)
    assert out[0, 0, 0, 0] == 1.0


def test_majority_vote_negative():
    """Test that majority of negative values gives -1."""
    key = jax.random.PRNGKey(0)
    x = jnp.array(
        [
            [
                [[1], [-1], [-1]],
                [[-1], [-1], [-1]],
                [[1], [-1], [-1]],
            ]
        ],
        dtype=jnp.float32,
    )

    pooling = MajorityPooling(kernel_size=3, strength=1.0, key=key, stride=3, padding_mode="edge")
    out = pooling(x)

    assert out.shape == (1, 1, 1, 1)
    assert out[0, 0, 0, 0] == -1.0


def test_majority_pooling_strength_scaling():
    """Test that strength parameter scales the output."""
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 3, 3, 1), dtype=jnp.float32)

    pooling = MajorityPooling(kernel_size=3, strength=2.5, key=key, stride=3, padding_mode="edge")
    out = pooling(x)

    assert jnp.allclose(out, 2.5)


def test_majority_pooling_multiple_channels():
    """Test pooling works with multiple channels."""
    key = jax.random.PRNGKey(0)
    x = jnp.ones((2, 6, 6, 3), dtype=jnp.float32)

    pooling = MajorityPooling(kernel_size=3, strength=1.0, key=key, stride=3, padding_mode="edge")
    out = pooling(x)

    assert out.shape == (2, 2, 2, 3)


# ConstantUnpooling tests


def test_basic_unpooling_no_unpad():
    """Test unpooling without unpadding."""
    x = jnp.array([[[[1], [-1]], [[1], [1]]]], dtype=jnp.float32)

    unpooling = ConstantUnpooling(kernel_size=3, strength=1.0)
    out = unpooling(x)

    assert out.shape == (1, 6, 6, 1)
    assert jnp.allclose(out[0, 0:3, 0:3, 0], 1.0)
    assert jnp.allclose(out[0, 0:3, 3:6, 0], -1.0)


def test_unpooling_with_unpad():
    """Test unpooling with unpadding to restore original shape."""
    x = jnp.array([[[[1]]]], dtype=jnp.float32)

    unpooling = ConstantUnpooling(kernel_size=3, strength=1.0, unpad=1)
    out = unpooling(x)

    assert out.shape == (1, 1, 1, 1)
    assert out[0, 0, 0, 0] == 1.0


def test_pool_unpool_roundtrip_shape():
    """Test that pooling followed by unpooling with unpad restores shape."""
    key = jax.random.PRNGKey(42)
    x = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(1, 5, 5, 2))

    pooling = MajorityPooling(kernel_size=3, strength=1.0, key=key, stride=3, padding_mode="edge")
    pooled = pooling(x)

    unpooling = ConstantUnpooling(kernel_size=3, strength=1.0, unpad=1)
    unpooled = unpooling(pooled)

    assert unpooled.shape[1] == 4 and unpooled.shape[2] == 4


def test_asymmetric_kernel():
    """Test unpooling with asymmetric kernel."""
    x = jnp.array([[[[1]]]], dtype=jnp.float32)

    unpooling = ConstantUnpooling(kernel_size=(2, 3), strength=1.0)
    out = unpooling(x)

    assert out.shape == (1, 2, 3, 1)


def test_asymmetric_unpad():
    """Test unpooling with asymmetric unpadding."""
    x = jnp.array([[[[1]]]], dtype=jnp.float32)

    unpooling = ConstantUnpooling(kernel_size=5, strength=1.0, unpad=(1, 2))
    out = unpooling(x)

    assert out.shape == (1, 3, 1, 1)


def test_unpooling_strength_scaling():
    """Test that strength parameter scales the output."""
    x = jnp.ones((1, 2, 2, 1), dtype=jnp.float32)

    unpooling = ConstantUnpooling(kernel_size=2, strength=3.0)
    out = unpooling(x)

    assert jnp.allclose(out, 3.0)


def test_constant_value_preservation():
    """Test that unpooling preserves constant values in blocks."""
    x = jnp.array([[[[5.0], [-3.0]]]], dtype=jnp.float32)

    unpooling = ConstantUnpooling(kernel_size=2, strength=1.0)
    out = unpooling(x)

    assert jnp.allclose(out[0, :, 0:2, 0], 5.0)
    assert jnp.allclose(out[0, :, 2:4, 0], -3.0)


# GlobalMajorityPooling tests


def test_global_pool_spatial_dimensions():
    """Test pooling over spatial dimensions."""
    x = jnp.array(
        [
            [
                [[1, -1], [1, 1]],
                [[-1, 1], [1, 1]],
            ]
        ],
        dtype=jnp.float32,
    )

    pooling = GlobalMajorityPooling(strength=1.0, axis=(1, 2))
    out = pooling(x)

    assert out.shape == (1, 2)
    assert jnp.allclose(out, jnp.array([[1.0, 1.0]]))


def test_global_pool_channel_dimension():
    """Test pooling over channel dimension."""
    x = jnp.array(
        [
            [
                [[1, -1, 1]],
            ]
        ],
        dtype=jnp.float32,
    )

    pooling = GlobalMajorityPooling(strength=1.0, axis=3)
    out = pooling(x)

    assert out.shape == (1, 1, 1)
    assert out[0, 0, 0] == 1.0


def test_global_pool_negative_majority():
    """Test that negative majority gives -1."""
    x = jnp.array([[[[1, -1, -1, -1]]]], dtype=jnp.float32)

    pooling = GlobalMajorityPooling(strength=1.0, axis=-1)
    out = pooling(x)

    assert out[0, 0, 0] == -1.0


def test_global_pool_tie_gives_negative():
    """Test that ties result in -1."""
    x = jnp.array([[[[1, -1, 1, -1]]]], dtype=jnp.float32)

    pooling = GlobalMajorityPooling(strength=1.0, axis=-1)
    out = pooling(x)

    assert out[0, 0, 0] == -1.0


def test_global_pool_strength_scaling():
    """Test strength parameter."""
    x = jnp.ones((1, 3, 3, 1), dtype=jnp.float32)

    pooling = GlobalMajorityPooling(strength=2.0, axis=(1, 2))
    out = pooling(x)

    assert jnp.allclose(out, 2.0)


# GlobalUnpooling tests


def test_global_unpool_expand_single_axis():
    """Test expanding along a single axis."""
    x = jnp.array([[[[1, -1]]]], dtype=jnp.float32)

    unpooling = GlobalUnpooling(strength=1.0, axis=2)
    out = unpooling(x)

    assert out.shape == (1, 1, 1, 1, 2)
    assert jnp.allclose(out[0, 0, 0, 0, :], jnp.array([1.0, -1.0]))


def test_global_unpool_expand_negative_axis():
    """Test expanding with negative axis index."""
    x = jnp.array([[[1, -1]]], dtype=jnp.float32)

    unpooling = GlobalUnpooling(strength=1.0, axis=-1)
    out = unpooling(x)

    assert out.shape == (1, 1, 2, 1)


def test_global_unpool_strength_scaling():
    """Test strength parameter."""
    x = jnp.ones((2, 3), dtype=jnp.float32)

    unpooling = GlobalUnpooling(strength=5.0, axis=1)
    out = unpooling(x)

    assert jnp.allclose(out, 5.0)


def test_global_unpool_value_preservation():
    """Test that values are preserved after expansion."""
    x = jnp.array([[[2.0, -3.0]]], dtype=jnp.float32)

    unpooling = GlobalUnpooling(strength=1.0, axis=0)
    out = unpooling(x)

    assert out.shape == (1, 1, 1, 2)
    assert jnp.allclose(out[0, 0, 0, :], jnp.array([2.0, -3.0]))


# Backward method tests


def test_majority_pooling_backward():
    """Test MajorityPooling backward returns zeros."""
    key = jax.random.PRNGKey(0)
    pooling = MajorityPooling(kernel_size=3, strength=1.0, key=key, stride=1, padding_mode="edge")

    x = jnp.ones((1, 3, 3, 1))
    y = jnp.ones((1, 3, 3, 1))
    y_hat = jnp.ones((1, 3, 3, 1))

    update = pooling.backward(x, y, y_hat)

    assert jnp.allclose(update.strength, 0.0)


def test_constant_unpooling_backward():
    """Test ConstantUnpooling backward returns zeros."""
    unpooling = ConstantUnpooling(kernel_size=2, strength=1.0)

    x = jnp.ones((1, 2, 2, 1))
    y = jnp.ones((1, 4, 4, 1))
    y_hat = jnp.ones((1, 4, 4, 1))

    update = unpooling.backward(x, y, y_hat)

    assert jnp.allclose(update.strength, 0.0)


def test_global_majority_pooling_backward():
    """Test GlobalMajorityPooling backward returns zeros."""
    pooling = GlobalMajorityPooling(strength=1.0, axis=(1, 2))

    x = jnp.ones((1, 3, 3, 1))
    y = jnp.ones((1, 1))
    y_hat = jnp.ones((1, 1))

    update = pooling.backward(x, y, y_hat)

    assert jnp.allclose(update.strength, 0.0)


def test_global_unpooling_backward():
    """Test GlobalUnpooling backward returns zeros."""
    unpooling = GlobalUnpooling(strength=1.0, axis=1)

    x = jnp.ones((1, 3))
    y = jnp.ones((1, 1, 3))
    y_hat = jnp.ones((1, 1, 3))

    update = unpooling.backward(x, y, y_hat)

    assert jnp.allclose(update.strength, 0.0)


# Integration tests


def test_pool_unpool_pipeline():
    """Test a complete pool->unpool pipeline."""
    key = jax.random.PRNGKey(123)

    x = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(1, 9, 9, 1))

    pooling = MajorityPooling(kernel_size=3, strength=1.0, key=key, stride=3, padding_mode="edge")
    pooled = pooling(x)
    assert pooled.shape == (1, 3, 3, 1)

    unpooling = ConstantUnpooling(kernel_size=3, strength=1.0, unpad=0)
    unpooled = unpooling(pooled)
    assert unpooled.shape == (1, 9, 9, 1)


def test_pool_with_padding_unpool_with_unpad():
    """Test pooling with padding and unpooling with unpad to restore shape."""
    key = jax.random.PRNGKey(42)
    x = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(1, 7, 7, 1))

    pooling = MajorityPooling(kernel_size=3, strength=1.0, key=key, stride=3, padding_mode="edge")
    pooled = pooling(x)

    unpooling = ConstantUnpooling(kernel_size=3, strength=1.0, unpad=1)
    unpooled = unpooling(pooled)

    assert unpooled.shape[1] >= 5 and unpooled.shape[2] >= 5
