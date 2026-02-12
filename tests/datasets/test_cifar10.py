import jax
import jax.numpy as jnp

from darnax.datasets.classification.cifar10 import Cifar10


def test_cifar10_generate_and_encode():
    """Generate projection, encode labels, and compute bounds for Cifar10."""
    key = jax.random.PRNGKey(1)
    w = Cifar10._generate_random_projection(key, 8, Cifar10.FLAT_DIM)
    assert w.shape == (8, Cifar10.FLAT_DIM)
    ds = Cifar10(batch_size=5, label_mode="c-rescale")
    y = jnp.array([0, 1, 2], dtype=jnp.int32)
    y_enc = ds._encode_labels(y)
    assert y_enc.shape == (3, ds.NUM_CLASSES)
    bounds = ds._compute_bounds(11)
    assert len(bounds) == -(-11 // ds.batch_size)


def test_cifar10_iter_valid_raises_when_no_validation():
    """iter_valid should raise NotImplementedError if no validation split exists."""
    ds = Cifar10()
    try:
        next(ds.iter_valid())
        raise AssertionError("Expected NotImplementedError when no validation split")
    except NotImplementedError:
        pass


def test_cifar10_invalid_init_args():
    """Constructor rejects invalid parameters like batch_size and validation_fraction."""
    try:
        Cifar10(batch_size=1)
        raise AssertionError("Expected ValueError for batch_size <= 1")
    except ValueError:
        pass
    try:
        Cifar10(validation_fraction=1.0)
        raise AssertionError("Expected ValueError for invalid validation_fraction")
    except ValueError:
        pass


def test_cifar10_build_and_iterators():
    """Build Cifar10 with a tiny per-class sample and validate spec and iterators."""
    ds = Cifar10(
        batch_size=8,
        linear_projection=32,
        num_images_per_class=3,
        validation_fraction=0.2,
        label_mode="c-rescale",
        x_transform="tanh",
    )
    ds.build(jax.random.PRNGKey(1))
    spec = ds.spec()
    assert spec["num_classes"] == ds.NUM_CLASSES
    assert spec["x_shape"] == (ds.input_dim,)
    # training iterator
    xb, yb = next(iter(ds))
    assert xb.shape[1] == ds.input_dim
    assert yb.shape[1] == ds.NUM_CLASSES
    # test and validation
    assert len(list(ds.iter_test())) >= 0
    if ds.x_valid is not None:
        assert len(list(ds.iter_valid())) >= 0


def test_cifar10_rescaling_default_applies_divide255():
    """Default rescaling for CIFAR-10 should divide by 255 (DEFAULT_RESCALING=divide255)."""
    ds = Cifar10(batch_size=4, rescaling="default", x_transform="identity")
    x = jnp.array([[0.0, 127.5, 255.0]], dtype=jnp.float32)
    result = ds._apply_rescaling(x)
    expected = x / 255.0
    assert jnp.allclose(result, expected)


def test_cifar10_rescaling_null_no_change():
    """Null rescaling should not change data."""
    ds = Cifar10(batch_size=4, rescaling="null", x_transform="identity")
    x = jnp.array([[0.0, 128.0, 255.0]], dtype=jnp.float32)
    result = ds._apply_rescaling(x)
    assert jnp.allclose(result, x)


def test_cifar10_rescaling_divide255():
    """divide255 should divide by 255."""
    ds = Cifar10(batch_size=4, rescaling="divide255", x_transform="identity")
    x = jnp.array([[0.0, 127.5, 255.0]], dtype=jnp.float32)
    result = ds._apply_rescaling(x)
    expected = x / 255.0
    assert jnp.allclose(result, expected)


def test_cifar10_rescaling_standardize():
    """Standardize should produce mean~0, std~1."""
    ds = Cifar10(batch_size=4, rescaling="standardize", x_transform="identity")
    x = jax.random.normal(jax.random.PRNGKey(0), (100, 3072))
    result = ds._apply_rescaling(x)
    assert jnp.isclose(result.mean(), 0.0, atol=1e-5)
    assert jnp.isclose(result.std(), 1.0, atol=1e-5)
