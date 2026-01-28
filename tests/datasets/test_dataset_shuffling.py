import jax
import jax.numpy as jnp

from darnax.datasets.classification.mnist import Mnist


def _flatten_batches(batches: list[tuple[jax.Array, jax.Array]]) -> tuple[jax.Array, jax.Array]:
    xs = jnp.concatenate([x for x, _ in batches], axis=0)
    ys = jnp.concatenate([y for _, y in batches], axis=0)
    return xs, ys


def test_shuffle_disabled_preserves_order() -> None:
    ds = Mnist(batch_size=4, linear_projection=None, flatten=True, shuffle=False)

    n = 20
    ds.x_train = jnp.arange(n, dtype=jnp.float32).reshape(n, 1)
    ds.y_train = jnp.arange(n, dtype=jnp.float32).reshape(n, 1)
    ds._train_bounds = ds._compute_bounds(n)

    xs, ys = _flatten_batches(list(iter(ds)))
    assert jnp.array_equal(xs[:, 0], jnp.arange(n, dtype=jnp.float32))
    assert jnp.array_equal(ys[:, 0], jnp.arange(n, dtype=jnp.float32))


def test_shuffle_each_epoch_changes_order_deterministically() -> None:
    n = 40

    ds1 = Mnist(batch_size=5, linear_projection=None, flatten=True, shuffle=True, shuffle_each_epoch=True)
    ds1.x_train = jnp.arange(n, dtype=jnp.float32).reshape(n, 1)
    ds1.y_train = jnp.arange(n, dtype=jnp.float32).reshape(n, 1)
    ds1._train_bounds = ds1._compute_bounds(n)
    ds1._train_epoch_key = jax.random.PRNGKey(0)

    # Epoch 1
    xs1_e1, ys1_e1 = _flatten_batches(list(iter(ds1)))
    assert jnp.array_equal(jnp.sort(xs1_e1[:, 0]), jnp.arange(n, dtype=jnp.float32))
    assert jnp.array_equal(xs1_e1[:, 0], ys1_e1[:, 0])

    # Epoch 2 (should differ from epoch 1 for a fixed initial key)
    xs1_e2, _ = _flatten_batches(list(iter(ds1)))
    assert not jnp.array_equal(xs1_e1[:, 0], xs1_e2[:, 0])

    # Determinism check: a fresh dataset with the same seed should match epoch 1.
    ds2 = Mnist(batch_size=5, linear_projection=None, flatten=True, shuffle=True, shuffle_each_epoch=True)
    ds2.x_train = jnp.arange(n, dtype=jnp.float32).reshape(n, 1)
    ds2.y_train = jnp.arange(n, dtype=jnp.float32).reshape(n, 1)
    ds2._train_bounds = ds2._compute_bounds(n)
    ds2._train_epoch_key = jax.random.PRNGKey(0)

    xs2_e1, _ = _flatten_batches(list(iter(ds2)))
    assert jnp.array_equal(xs1_e1[:, 0], xs2_e1[:, 0])
