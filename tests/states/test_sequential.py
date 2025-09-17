import jax
import jax.numpy as jnp
import pytest

from bionet.states.sequential import SequentialState

# -------------------------
# Construction & basics
# -------------------------


def test_init_valid_shapes_and_defaults():
    """Check valid shapes at init."""
    sizes = [5, 3, 2, (3, 3)]
    s = SequentialState(sizes=sizes, dtype=jnp.float32)
    assert len(s.states) == len(sizes) == len(s)
    assert s.states[0].shape == (1, 5)
    assert s.states[1].shape == (1, 3)
    assert s.states[2].shape == (1, 2)
    assert s.states[3].shape == (1, 3, 3)
    assert s.states[0].dtype == jnp.float32


@pytest.mark.parametrize(
    "bad_sizes",
    [
        [0, 3],  # zero not allowed
        [-1, 2, 3],  # negative not allowed
        [3, 2.5, 1],  # non-int
        [3, (-1,), 1],  # non-int
        [3, (3.4), 1],  # non-int
        [],  # empty sequence
    ],
)
def test_init_rejects_bad_sizes(bad_sizes):
    """Reject creation if size array is malformed."""
    with pytest.raises(AssertionError):
        SequentialState(sizes=bad_sizes)


def test_getitem_reads_layers():
    """Check correct shape handling."""
    s = SequentialState([2, 4, 3])
    x = s[1]
    assert x.shape == (1, 4)


def test_replace_overwrites_entire_state_functionally():
    """Check functional update in replace."""
    s = SequentialState([2, 3])
    val = 2.0
    new_states = [jnp.ones((7, 2)), jnp.full((7, 3), val)]
    s2 = s.replace(new_states)

    # original untouched
    assert s.states[0].shape == (1, 2)
    # new one changed
    assert s2.states[0].shape == (7, 2)
    assert jnp.all(s2.states[0] == 1.0)
    assert jnp.all(s2.states[1] == val)


def test_replace_val_writes_single_index_functionally():
    """Check single updateis functional."""
    s = SequentialState([2, 3, 4])
    s2 = s.replace_val(1, jnp.ones((5, 3)))
    # original untouched
    assert s.states[1].shape == (1, 3)
    # updated copy
    assert s2.states[1].shape == (5, 3)
    assert jnp.all(s2.states[1] == 1.0)
    # others unchanged
    assert s2.states[0].shape == (1, 2)
    assert s2.states[2].shape == (1, 4)


def test_init_sets_input_and_optional_output():
    """Checks correct setting of input and output."""
    s = SequentialState([3, 5])
    x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    s2 = s.init(x)
    assert s2.states[0].shape == (2, 3)
    assert jnp.all(s2.states[0] == x)
    # output stays zero when y=None
    assert s2.states[1].shape == (2, 5)
    assert jnp.all(s2.states[1] == 0.0)

    y = jnp.arange(10, dtype=jnp.float32).reshape(2, 5)
    s3 = s.init(x, y)
    assert jnp.all(s3.states[0] == x)
    assert jnp.all(s3.states[1] == y)


def test_init_shape_checks():
    """Checks various situations of bad ranks."""
    s = SequentialState([3, 4])
    x_bad_rank = jnp.zeros((3,))  # rank-1
    with pytest.raises(AssertionError):
        s.init(x_bad_rank)

    x = jnp.zeros((2, 3))
    y_bad_rank = jnp.zeros((4,))  # rank-1
    with pytest.raises(AssertionError):
        s.init(x, y_bad_rank)

    x_bad_feat = jnp.zeros((2, 5))
    with pytest.raises(AssertionError):
        s.init(x_bad_feat)

    y_bad_feat = jnp.zeros((2, 5))
    with pytest.raises(AssertionError):
        s.init(x, y_bad_feat)

    y_bad_batch = jnp.zeros((3, 4))
    with pytest.raises(AssertionError):
        s.init(x, y_bad_batch)


def test_pytree_and_static_dtype():
    """Check if dtype is static."""
    # Ensure object is a PyTree and dtype is static (not in leaves).
    s = SequentialState([2, 2], dtype=jnp.float32)
    leaves, _ = jax.tree.flatten(s)
    # leaves contains only array leaves; dtype is static and not present there.
    assert hasattr(s, "dtype")
    # sanity: the partitioned "leaves" are arrays from states only
    assert all(isinstance(x, jnp.ndarray) for x in leaves)


# -------------------------
# JIT smoke tests
# -------------------------


def test_jit_init_basic():
    """`init` should run under jax.jit and produce the expected shapes/values."""
    s = SequentialState([3, 5], dtype=jnp.float32)

    @jax.jit
    def jitted_init(st, x, y=None):
        # NOTE: calling the bound method via the class to keep the function pure
        return SequentialState.init(st, x, y)

    x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    y = jnp.arange(10, dtype=jnp.float32).reshape(2, 5)

    s2 = jitted_init(s, x, None)
    assert s2.states[0].shape == (2, 3)
    assert s2.states[1].shape == (2, 5)
    assert jnp.all(s2.states[0] == x)
    assert jnp.all(s2.states[1] == 0.0)

    s3 = jitted_init(s, x, y)
    assert jnp.all(s3.states[0] == x)
    assert jnp.all(s3.states[1] == y)


def test_jit_replace_val():
    """`replace_val` should be usable inside a jitted function."""
    s = SequentialState([2, 3, 4])

    @jax.jit
    def update_mid(st, val):
        return SequentialState.replace_val(st, 1, val)

    val = jnp.ones((5, 3), dtype=s.dtype)
    s2 = update_mid(s, val)
    assert s2.states[1].shape == (5, 3)
    assert jnp.all(s2.states[1] == 1.0)
    # others unchanged in shape
    assert s2.states[0].shape == (1, 2)
    assert s2.states[2].shape == (1, 4)


def test_jit_reads_and_computes():
    """Read from the state under JIT and perform a simple computation."""
    s = SequentialState([3, 2])
    x = jnp.ones((4, 3), dtype=s.dtype)
    s2 = s.init(x)  # non-jitted preparation is fine here

    @jax.jit
    def compute_sum(st):
        return jnp.sum(st[0])  # relies on __getitem__

    out = compute_sum(s2)
    # 4*3 ones
    assert out == pytest.approx(12.0)


def test_jit_batch_size_changes_allowed():
    """Changing batch size should recompile but still run correctly."""
    s = SequentialState([3, 5], dtype=jnp.float32)

    @jax.jit
    def jitted_init(st, x, y=None):
        return SequentialState.init(st, x, y)

    # First batch size
    x1 = jnp.zeros((2, 3), dtype=jnp.float32)
    s2 = jitted_init(s, x1, None)
    assert s2.states[0].shape == (2, 3)
    assert s2.states[1].shape == (2, 5)

    # Different batch size — expect retrace/recompile under the hood but correctness here
    x2 = jnp.zeros((5, 3), dtype=jnp.float32)
    s3 = jitted_init(s, x2, None)
    assert s3.states[0].shape == (5, 3)
    assert s3.states[1].shape == (5, 5)


def test_plain_jax_jit_whole_object():
    """Simple jit that avoids partition, for now."""
    s = SequentialState([(3,), (5,)], dtype=jnp.float32)

    @jax.jit
    def run(st, x):
        return SequentialState.init(st, x)

    out = run(s, jnp.zeros((2, 3), dtype=jnp.float32))
    assert out[0].shape == (2, 3)
    assert out[1].shape == (2, 5)
