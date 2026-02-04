
import jax.numpy as jnp
import pytest
import jax
from darnax.datasets.classification.mnist import Mnist
from darnax.datasets.classification.cifar10 import Cifar10
from darnax.datasets.classification.fashion_mnist import FashionMnist
from darnax.datasets.classification.cifar10_features import Cifar10FeaturesSmall

@pytest.mark.parametrize("dataset_cls", [Mnist, Cifar10, FashionMnist])
def test_rescale_flag_images(dataset_cls):
    """Test that rescale=True yields [0, 1] and rescale=False yields [0, 255]."""
    key = jax.random.PRNGKey(0)

    # Test rescale=True (default)
    ds_true = dataset_cls(batch_size=4, shuffle=False, rescale=True, x_transform="identity", linear_projection=None)
    ds_true.build(key)
    x_true, _ = next(iter(ds_true))
    
    assert jnp.min(x_true) >= 0.0
    assert jnp.max(x_true) <= 1.0
    # Sanity check: ensure it's not all zeros (unless dataset is weird, but these are standard)
    assert jnp.max(x_true) > 0.0

    # Test rescale=False
    ds_false = dataset_cls(batch_size=4, shuffle=False, rescale=False, x_transform="identity", linear_projection=None)
    ds_false.build(key)
    x_false, _ = next(iter(ds_false))
    
    assert jnp.min(x_false) >= 0.0
    # Should contain values > 1.0 (approaching 255)
    assert jnp.max(x_false) > 1.0
    assert jnp.max(x_false) <= 255.0

def test_rescale_flag_features():
    """Test that Cifar10Features accepts the rescale flag (even if no-op)."""
    key = jax.random.PRNGKey(0)
    
    # Just ensure it doesn't crash and returns data
    ds = Cifar10FeaturesSmall(batch_size=4, rescale=True)
    ds.build(key)
    x, _ = next(iter(ds))
    assert x.shape == (4, 512)

    ds = Cifar10FeaturesSmall(batch_size=4, rescale=False)
    ds.build(key)
    x, _ = next(iter(ds))
    assert x.shape == (4, 512)
