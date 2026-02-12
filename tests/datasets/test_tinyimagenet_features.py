import jax
import jax.numpy as jnp

from darnax.datasets.classification.tinyimagenet_features import TinyImagenetFeatures


def test_tinyimagenet_projection_transform_and_subsample():
    """Apply projection, x-transform and subsampling for TinyImageNet features."""
    ds = TinyImagenetFeatures(batch_size=4, x_transform="sign", linear_projection=16)
    key = jax.random.PRNGKey(2)
    w = ds._generate_random_projection(key, 16, ds.FEAT_DIM)
    x = jnp.zeros((3, ds.FEAT_DIM), dtype=jnp.float32)
    x_proj = ds._apply_projection(w, x)
    assert x_proj.shape == (3, 16)
    x_tr = ds._apply_x_transform(x_proj)
    assert jnp.all(x_tr == -1.0)
    # create two examples per class (use first 10 classes for simplicity)
    y = jnp.repeat(jnp.arange(10, dtype=jnp.int32), 2)
    x_many = jnp.vstack(
        [jnp.full((ds.FEAT_DIM,), float(i), dtype=jnp.float32) for i in range(10) for _ in range(2)]
    )
    x_sub, y_sub = ds._subsample_per_class(key, x_many, y, k=1)
    assert x_sub.shape[0] == 10
    assert x_sub.shape[1] == ds.FEAT_DIM
    y_enc = ds._encode_labels(y_sub)
    assert y_enc.shape[1] == TinyImagenetFeatures.NUM_CLASSES


def test_tinyimagenet_invalid_init_args():
    """Constructor rejects invalid args."""
    try:
        TinyImagenetFeatures(batch_size=1)
        raise AssertionError("Expected ValueError for batch_size <= 1")
    except ValueError:
        pass
    try:
        TinyImagenetFeatures(linear_projection=0)
        raise AssertionError("Expected ValueError for invalid linear_projection")
    except ValueError:
        pass


def test_tinyimagenet_build_and_spec_and_iterators():
    """Build TinyImageNet features dataset with a small per-class cap and validate spec/iterators."""
    ds = TinyImagenetFeatures(
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


def test_tinyimagenet_rescaling_default_no_change():
    """Default rescaling for features should not change data (DEFAULT_RESCALING="null")."""
    ds = TinyImagenetFeatures(batch_size=4, rescaling="default")
    x = jnp.array([[0.0, 0.5, 1.0]], dtype=jnp.float32)
    result = ds._apply_rescaling(x)
    assert jnp.allclose(result, x)


def test_tinyimagenet_rescaling_null_no_change():
    """Null rescaling should not change data."""
    ds = TinyImagenetFeatures(batch_size=4, rescaling="null")
    x = jnp.array([[0.0, 0.5, 1.0]], dtype=jnp.float32)
    result = ds._apply_rescaling(x)
    assert jnp.allclose(result, x)


def test_tinyimagenet_rescaling_divide255():
    """divide255 should divide by 255."""
    ds = TinyImagenetFeatures(batch_size=4, rescaling="divide255")
    x = jnp.array([[0.0, 127.5, 255.0]], dtype=jnp.float32)
    result = ds._apply_rescaling(x)
    expected = x / 255.0
    assert jnp.allclose(result, expected)


def test_tinyimagenet_rescaling_standardize():
    """Standardize should produce mean~0, std~1."""
    ds = TinyImagenetFeatures(batch_size=4, rescaling="standardize")
    x = jax.random.normal(jax.random.PRNGKey(0), (100, 512)) * 3 + 5
    result = ds._apply_rescaling(x)
    assert jnp.isclose(result.mean(), 0.0, atol=1e-5)
    assert jnp.isclose(result.std(), 1.0, atol=1e-5)
