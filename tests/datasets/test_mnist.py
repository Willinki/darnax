import jax
import jax.numpy as jnp

from darnax.datasets.classification.mnist import Mnist


def test_mnist_generate_random_projection():
    """Generate a random projection and check its shape and dtype."""
    key = jax.random.PRNGKey(0)
    w = Mnist._generate_random_projection(key, 5, Mnist.FLAT_DIM)
    assert w.shape == (5, Mnist.FLAT_DIM)
    assert w.dtype == jnp.float32


def test_mnist_preprocess_and_encode_labels():
    """Preprocess with sign transform and encode labels in pm1 mode."""
    ds = Mnist(batch_size=4, linear_projection=None, x_transform="sign", label_mode="pm1", rescaling="null")
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


def test_mnist_invalid_init_and_len_error():
    """Constructor rejects bad args and __len__ errors when not built."""
    try:
        Mnist(batch_size=1)
        raise AssertionError("Expected ValueError for batch_size <= 1")
    except ValueError:
        pass
    ds = Mnist()
    try:
        _ = len(ds)
        raise AssertionError("Expected RuntimeError from __len__ when not built")
    except RuntimeError:
        pass


def test_mnist_build_iterators_and_spec():
    """Build MNIST with small per-class subsample and validate iterators and spec."""
    ds = Mnist(
        batch_size=4,
        linear_projection=16,
        num_images_per_class=4,
        validation_fraction=0.25,
        label_mode="ooe",
        x_transform="identity",
    )
    ds.build(jax.random.PRNGKey(0))
    spec = ds.spec()
    assert spec["num_classes"] == ds.NUM_CLASSES
    assert spec["x_shape"] == (ds.input_dim,)
    # length and iterator
    n_batches = len(ds)
    assert n_batches > 0
    xb, yb = next(iter(ds))
    assert xb.shape[1] == ds.input_dim
    assert yb.shape[1] == ds.NUM_CLASSES
    # test iterator
    tb = list(ds.iter_test())
    assert all(x.shape[1] == ds.input_dim for x, _ in tb)
    # validation
    if ds.x_valid is not None:
        vb = list(ds.iter_valid())
        assert all(x.shape[1] == ds.input_dim for x, _ in vb)


def test_mnist_rescaling_default_applies_divide255():
    """Default rescaling for MNIST should divide by 255 (DEFAULT_RESCALING=divide255)."""
    ds = Mnist(batch_size=4, rescaling="default", x_transform="identity")
    x = jnp.array([[0.0, 127.5, 255.0]], dtype=jnp.float32)
    result = ds._apply_rescaling(x)
    expected = x / 255.0
    assert jnp.allclose(result, expected)


def test_mnist_rescaling_null_no_change():
    """Null rescaling should not change data."""
    ds = Mnist(batch_size=4, rescaling="null", x_transform="identity")
    x = jnp.array([[0.0, 128.0, 255.0]], dtype=jnp.float32)
    result = ds._apply_rescaling(x)
    assert jnp.allclose(result, x)


def test_mnist_rescaling_divide255():
    """divide255 should divide by 255."""
    ds = Mnist(batch_size=4, rescaling="divide255", x_transform="identity")
    x = jnp.array([[0.0, 127.5, 255.0]], dtype=jnp.float32)
    result = ds._apply_rescaling(x)
    expected = x / 255.0
    assert jnp.allclose(result, expected)


def test_mnist_rescaling_standardize():
    """Standardize should produce mean~0, std~1."""
    ds = Mnist(batch_size=4, rescaling="standardize", x_transform="identity")
    x = jax.random.normal(jax.random.PRNGKey(0), (100, 784))
    result = ds._apply_rescaling(x)
    assert jnp.isclose(result.mean(), 0.0, atol=1e-5)
    assert jnp.isclose(result.std(), 1.0, atol=1e-5)
