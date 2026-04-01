"""Tests for BlockSparseMatrix pytree and block_sparse_matvec."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darnax.utils.block_sparse_matmul import block_sparse_matvec, make_block_sparse

KEY = jax.random.key(0)


# ---------- Fixtures ----------


@pytest.fixture
def mat():
    """16x16 block-sparse matrix, b=4, 12 active blocks (avg=3.0 > er_safety_factor)."""
    return make_block_sparse(KEY, m=16, n=16, b=4, n_active_blocks=12)


# ---------- make_block_sparse ----------


def test_make_block_sparse_shape(mat):
    """BlockSparseMatrix has the expected shape and block size."""
    assert mat.shape == (16, 16)
    assert mat.b == 4
    assert mat.values.shape == (12, 4, 4)


def test_make_block_sparse_n_active(mat):
    """n_active equals the requested number of active blocks."""
    assert mat.n_active == 12


def test_make_block_sparse_block_idx_range(mat):
    """Block row/col indices are within valid grid bounds."""
    k_row = 16 // 4
    k_col = 16 // 4
    assert int(mat.block_row_idx.min()) >= 0
    assert int(mat.block_row_idx.max()) < k_row
    assert int(mat.block_col_idx.min()) >= 0
    assert int(mat.block_col_idx.max()) < k_col


def test_make_block_sparse_no_duplicate_blocks(mat):
    """All (row, col) block positions are unique."""
    flat = np.asarray(mat.block_row_idx) * (16 // 4) + np.asarray(mat.block_col_idx)
    assert len(flat) == len(set(flat.tolist()))


def test_make_block_sparse_dtype_default(mat):
    """Default dtype is float32."""
    assert mat.dtype == jnp.float32
    assert mat.values.dtype == jnp.float32


def test_make_block_sparse_dtype_float16():
    """Requested dtype float16 is propagated to values."""
    m = make_block_sparse(KEY, m=16, n=16, b=4, n_active_blocks=12, dtype=jnp.float16)
    assert m.dtype == jnp.float16
    assert m.values.dtype == jnp.float16


def test_make_block_sparse_raises_non_divisible_m():
    """ValueError when m is not divisible by b."""
    with pytest.raises(ValueError, match="m=9"):
        make_block_sparse(KEY, m=9, n=8, b=4, n_active_blocks=2)


def test_make_block_sparse_raises_non_divisible_n():
    """ValueError when n is not divisible by b."""
    with pytest.raises(ValueError, match="n=10"):
        make_block_sparse(KEY, m=8, n=10, b=4, n_active_blocks=2)


def test_make_block_sparse_raises_too_many_blocks():
    """ValueError when n_active_blocks exceeds total available blocks."""
    # k_row=2, k_col=2 → 4 total slots
    with pytest.raises(ValueError, match="n_active_blocks=5"):
        make_block_sparse(KEY, m=8, n=8, b=4, n_active_blocks=5)


def test_make_block_sparse_er_warning():
    """UserWarning when avg degree is too low to guarantee giant component."""
    # k_row=2, n_active=2 → avg=1.0 <= er_safety_factor=2.0
    with pytest.warns(UserWarning, match="Giant component"):
        make_block_sparse(KEY, m=8, n=8, b=4, n_active_blocks=2)


def test_make_block_sparse_no_er_warning():
    """No UserWarning when avg degree is safely above the ER threshold."""
    # k_row=4, n_active=12 → avg=3.0 > 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_block_sparse(KEY, m=16, n=16, b=4, n_active_blocks=12)


# ---------- PyTree ----------


def test_pytree_single_leaf(mat):
    """Flattening a BlockSparseMatrix yields exactly one leaf (values)."""
    leaves, _ = jax.tree_util.tree_flatten(mat)
    assert len(leaves) == 1
    assert jnp.allclose(leaves[0], mat.values)


def test_pytree_roundtrip(mat):
    """Flatten/unflatten roundtrip preserves all fields."""
    leaves, treedef = jax.tree_util.tree_flatten(mat)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.allclose(restored.values, mat.values)
    assert restored.shape == mat.shape
    assert restored.b == mat.b
    assert restored.dtype == mat.dtype
    assert jnp.array_equal(restored.block_row_idx, mat.block_row_idx)
    assert jnp.array_equal(restored.block_col_idx, mat.block_col_idx)


# ---------- block_sparse_matvec ----------


def test_matvec_matches_dense():
    """block_sparse_matvec matches the equivalent dense matrix multiply."""
    key = jax.random.key(1)
    m, n, b = 16, 16, 4
    mat = make_block_sparse(key, m=m, n=n, b=b, n_active_blocks=12)

    # Build the equivalent dense matrix
    dense = jnp.zeros((m, n))
    for k in range(mat.n_active):
        br = int(mat.block_row_idx[k])
        bc = int(mat.block_col_idx[k])
        dense = dense.at[br * b : (br + 1) * b, bc * b : (bc + 1) * b].set(mat.values[k])

    x = jax.random.normal(jax.random.key(2), (n,))
    assert jnp.allclose(block_sparse_matvec(mat, x), dense @ x, atol=1e-5)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_matvec_matches_dense_random(seed):
    """block_sparse_matvec matches dense for random seeds."""
    key = jax.random.key(seed)
    m, n, b = 16, 12, 4
    mat = make_block_sparse(key, m=m, n=n, b=b, n_active_blocks=6)

    dense = jnp.zeros((m, n))
    for k in range(mat.n_active):
        br = int(mat.block_row_idx[k])
        bc = int(mat.block_col_idx[k])
        dense = dense.at[br * b : (br + 1) * b, bc * b : (bc + 1) * b].set(mat.values[k])

    x = jax.random.normal(jax.random.key(seed + 100), (n,))
    assert jnp.allclose(block_sparse_matvec(mat, x), dense @ x, atol=1e-5)


def test_matmul_operator(mat):
    """The @ operator dispatches to block_sparse_matvec."""
    x = jax.random.normal(KEY, (mat.shape[1],))
    assert jnp.allclose(mat @ x, block_sparse_matvec(mat, x))


def test_matvec_jit(mat):
    """block_sparse_matvec produces the same result under jit."""
    x = jax.random.normal(KEY, (mat.shape[1],))
    eager = block_sparse_matvec(mat, x)
    jitted = jax.jit(block_sparse_matvec)(mat, x)
    assert jnp.allclose(eager, jitted)


def test_matvec_grad_through_values(mat):
    """Gradients flow through BlockSparseMatrix.values."""
    x = jax.random.normal(KEY, (mat.shape[1],))

    def f(m):
        return block_sparse_matvec(m, x).sum()

    grads = jax.grad(f)(mat)
    assert grads.values.shape == mat.values.shape
    assert jnp.all(jnp.isfinite(grads.values))


def test_matvec_output_shape(mat):
    """Output shape is (m,) for a (m, n) matrix and (n,) input."""
    x = jax.random.normal(KEY, (mat.shape[1],))
    y = block_sparse_matvec(mat, x)
    assert y.shape == (mat.shape[0],)


def test_matvec_non_square():
    """block_sparse_matvec works for non-square matrices."""
    key = jax.random.key(3)
    mat = make_block_sparse(key, m=12, n=8, b=4, n_active_blocks=6)
    x = jax.random.normal(key, (8,))
    y = block_sparse_matvec(mat, x)
    assert y.shape == (12,)
