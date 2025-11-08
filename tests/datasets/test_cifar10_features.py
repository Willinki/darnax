import jax
import jax.numpy as jnp

from darnax.datasets.classification.cifar10_features import (
    Cifar10FeaturesLarge,
    Cifar10FeaturesSmall,
)


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
    x_sub, y_sub = ds._subsample_per_class(key, x_many, y, k=1)
    bsize = 10
    assert x_sub.shape[0] == bsize
    assert x_sub.shape[1] == ds.FEAT_DIM
    y_enc = ds._encode_labels(y_sub)
    assert y_enc.shape[1] == Cifar10FeaturesSmall.NUM_CLASSES


def test_cifar_invalid_init_args_and_large_inheritance():
    """Constructor rejects invalid args and Large variant inherits constants."""
    try:
        Cifar10FeaturesSmall(batch_size=1)
        raise AssertionError("Expected ValueError for batch_size <= 1")
    except ValueError:
        pass
    try:
        Cifar10FeaturesSmall(linear_projection=0)
        raise AssertionError("Expected ValueError for invalid linear_projection")
    except ValueError:
        pass
    assert Cifar10FeaturesLarge.FEAT_DIM != Cifar10FeaturesSmall.FEAT_DIM
    assert Cifar10FeaturesLarge.NUM_CLASSES == Cifar10FeaturesSmall.NUM_CLASSES


def test_cifar_build_and_spec_and_iterators():
    """Build CIFAR-10 features dataset with a small per-class cap and validate spec/iterators."""
    ds = Cifar10FeaturesSmall(
        batch_size=4,
        linear_projection=8,
        num_images_per_class=3,
        validation_fraction=0.25,
        x_transform="identity",
    )
    ds.build(jax.random.PRNGKey(3))
    spec = ds.spec()
    assert spec["num_classes"] == ds.NUM_CLASSES
    assert spec["x_shape"] == (ds.input_dim,)
    # training iterator
    xb, yb = next(iter(ds))
    assert xb.shape[1] == ds.input_dim
    assert yb.shape[1] == ds.num_classes
    # test iterator
    assert len(list(ds.iter_test())) >= 0
    if ds.x_valid is not None:
        assert len(list(ds.iter_valid())) >= 0
