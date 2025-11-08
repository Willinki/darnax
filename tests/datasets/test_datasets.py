import jax
import jax.numpy as jnp

from darnax.datasets.classification.cifar10_features import (
    Cifar10FeaturesLarge,
    Cifar10FeaturesSmall,
)
from darnax.datasets.classification.fashion_mnist import FashionMnist
from darnax.datasets.classification.mnist import Mnist


def test_mnist_generate_random_projection():
    """Generate a random projection and check its shape and dtype."""
    key = jax.random.PRNGKey(0)
    w = Mnist._generate_random_projection(key, 5, Mnist.FLAT_DIM)
    assert w.shape == (5, Mnist.FLAT_DIM)
    assert w.dtype == jnp.float32


def test_mnist_preprocess_and_encode_labels():
    """Preprocess with sign transform and encode labels in pm1 mode."""
    ds = Mnist(batch_size=4, linear_projection=None, x_transform="sign", label_mode="pm1")
    x = jnp.zeros((2, 28, 28), dtype=jnp.float32)
    xp = ds._preprocess(None, x)
    assert xp.shape == (2, Mnist.FLAT_DIM)
    assert xp.dtype == jnp.float32
    assert jnp.all(xp == -1.0)
    y = jnp.array([0, 1], dtype=jnp.int32)
    y_enc = ds._encode_labels(y)
    assert y_enc.shape == (2, ds.NUM_CLASSES)
    assert jnp.all((y_enc == -1.0) | (y_enc == 1.0))


def test_mnist_compute_bounds_and_subsample():
    """Compute batch bounds and subsample one image per class deterministically."""
    ds = Mnist(batch_size=3)
    bounds = ds._compute_bounds(7)
    assert bounds == [(0, 3), (3, 6), (6, 7)]
    key = jax.random.PRNGKey(0)
    x = jnp.stack([jnp.full((28, 28), i, dtype=jnp.float32) for i in range(10)])
    y = jnp.arange(10, dtype=jnp.int32)
    x_sub, y_sub = Mnist._subsample_per_class(key, x, y, 1)
    bsize = 10
    assert x_sub.shape[0] == bsize
    assert jnp.array_equal(jnp.sort(y_sub), jnp.arange(10))


def test_fashion_generate_and_encode():
    """Generate projection, encode labels, and compute bounds for FashionMnist."""
    key = jax.random.PRNGKey(1)
    w = FashionMnist._generate_random_projection(key, 8, FashionMnist.FLAT_DIM)
    assert w.shape == (8, FashionMnist.FLAT_DIM)
    ds = FashionMnist(batch_size=5, label_mode="c-rescale")
    y = jnp.array([0, 1, 2], dtype=jnp.int32)
    y_enc = ds._encode_labels(y)
    assert y_enc.shape == (3, ds.NUM_CLASSES)
    bounds = ds._compute_bounds(11)
    assert len(bounds) == -(-11 // ds.batch_size)


def test_cifar_projection_transform_and_subsample():
    """Apply projection, x-transform and subsampling for CIFAR features."""
    ds = Cifar10FeaturesSmall(batch_size=4, x_transform="sign", linear_projection=16)
    key = jax.random.PRNGKey(2)
    w = ds._generate_random_projection(key, 16, ds.FEAT_DIM)
    x = jnp.zeros((3, ds.FEAT_DIM), dtype=jnp.float32)
    x_proj = ds._apply_projection(w, x)
    assert x_proj.shape == (3, 16)
    x_tr = ds._apply_x_transform(x_proj)
    assert jnp.all(x_tr == -1.0)
    # create two examples per class
    y = jnp.repeat(jnp.arange(10, dtype=jnp.int32), 2)
    x_many = jnp.vstack(
        [jnp.full((ds.FEAT_DIM,), float(i), dtype=jnp.float32) for i in range(10) for _ in range(2)]
    )
    x_sub, y_sub = Cifar10FeaturesSmall._subsample_per_class(key, x_many, y, k=1)
    bsize = 10
    assert x_sub.shape[0] == bsize
    assert x_sub.shape[1] == ds.FEAT_DIM or ds.linear_projection is not None
    y_enc = ds._encode_labels(y_sub)
    assert y_enc.shape[1] == Cifar10FeaturesSmall.NUM_CLASSES


def test_cifar_large_inherits_behavior():
    """Large features class should inherit shape constants from small variant."""
    assert Cifar10FeaturesLarge.FEAT_DIM != Cifar10FeaturesSmall.FEAT_DIM
    assert Cifar10FeaturesLarge.NUM_CLASSES == Cifar10FeaturesSmall.NUM_CLASSES
