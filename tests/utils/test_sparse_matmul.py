"""Tests for SparseMatrix pytree and sparse_matvec."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darnax.utils.sparse_matmul import sparse_matvec, to_sparse

# ---------- Fixtures ----------


@pytest.fixture
def simple_dense():
    """3x4 dense matrix with a few non-zeros."""
    # 3x4 matrix with a few non-zeros
    return jnp.array(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
            [0.0, 4.0, 0.0, 5.0],
        ]
    )


@pytest.fixture
def simple_sparse(simple_dense):
    """Sparse version of simple_dense."""
    return to_sparse(simple_dense)


# ---------- to_sparse ----------


def test_to_sparse_nnz(simple_dense, simple_sparse):
    """Nnz equals the count of non-zero entries in the dense matrix."""
    assert simple_sparse.nnz == int(jnp.sum(simple_dense != 0))


def test_to_sparse_shape(simple_dense, simple_sparse):
    """Shape is preserved after conversion to sparse."""
    assert simple_sparse.shape == simple_dense.shape


def test_to_sparse_dtype(simple_dense, simple_sparse):
    """Dtype and values.dtype match the source dense matrix."""
    assert simple_sparse.dtype == jnp.result_type(simple_dense)
    assert simple_sparse.values.dtype == simple_sparse.dtype


def test_to_sparse_values_match(simple_dense, simple_sparse):
    """Reconstructed dense matrix matches the original."""
    # Reconstruct dense from sparse and compare
    dense_rec = jnp.zeros(simple_sparse.shape, dtype=simple_sparse.dtype)
    dense_rec = dense_rec.at[
        np.asarray(simple_sparse.row_indices),
        np.asarray(simple_sparse.col_indices),
    ].set(simple_sparse.values)
    assert jnp.allclose(dense_rec, simple_dense)


def test_to_sparse_linear_indices_sorted(simple_sparse):
    """linear_indices are in non-decreasing (row-major) order."""
    lin = np.asarray(simple_sparse.linear_indices)
    assert np.all(lin[:-1] <= lin[1:]), "linear_indices must be sorted"


def test_to_sparse_tol_drops_small_entries():
    """Entries at or below tol are excluded from the sparse representation."""
    dense = jnp.array([[1e-9, 1.0], [0.5, 0.0]])
    sp = to_sparse(dense, tol=1e-6)
    assert sp.nnz == 2  # only 1.0 and 0.5 survive


def test_to_sparse_all_zeros():
    """All-zero matrix converts to a sparse matrix with nnz == 0."""
    dense = jnp.zeros((3, 3))
    sp = to_sparse(dense)
    assert sp.nnz == 0


def test_to_sparse_dense_matrix():
    """Fully dense matrix stores all n*m entries."""
    dense = jnp.ones((4, 4))
    sp = to_sparse(dense)
    assert sp.nnz == 16


# ---------- PyTree ----------


def test_pytree_roundtrip(simple_sparse):
    """Flatten/unflatten roundtrip preserves values, shape, and dtype."""
    leaves, treedef = jax.tree_util.tree_flatten(simple_sparse)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.allclose(restored.values, simple_sparse.values)
    assert restored.shape == simple_sparse.shape
    assert restored.dtype == simple_sparse.dtype


def test_pytree_single_leaf(simple_sparse):
    """SparseMatrix has exactly one dynamic leaf (values)."""
    leaves, _ = jax.tree_util.tree_flatten(simple_sparse)
    assert len(leaves) == 1
    assert jnp.allclose(leaves[0], simple_sparse.values)


def test_jit_matvec(simple_sparse):
    """JIT-compiled sparse_matvec matches the eager result."""
    x = jnp.ones(simple_sparse.shape[1])
    result = jax.jit(sparse_matvec)(simple_sparse, x)
    expected = sparse_matvec(simple_sparse, x)
    assert jnp.allclose(result, expected)


def test_grad_through_values(simple_sparse):
    """Gradient w.r.t. values equals x[col_indices]."""
    x = jnp.ones(simple_sparse.shape[1])

    def f(sp):
        return sparse_matvec(sp, x).sum()

    grads = jax.grad(f)(simple_sparse)
    # gradient w.r.t. values should be x[col_indices]
    expected_grad = x[simple_sparse.col_indices]
    assert jnp.allclose(grads.values, expected_grad)


# ---------- sparse_matvec ----------


def test_sparse_matvec_matches_dense(simple_dense, simple_sparse):
    """sparse_matvec result matches dense matrix multiply."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = sparse_matvec(simple_sparse, x)
    expected = simple_dense @ x
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_sparse_matvec_random(seed):
    """sparse_matvec matches dense @ x on random matrices across seeds."""
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    dense = jax.random.normal(k1, (8, 12))
    mask = jax.random.bernoulli(k2, 0.3, (8, 12))
    dense = dense * mask
    x = jax.random.normal(k3, (12,))

    sp = to_sparse(dense)
    assert jnp.allclose(sparse_matvec(sp, x), dense @ x, atol=1e-5)


# ---------- @ operator ----------


def test_matmul_operator(simple_dense, simple_sparse):
    """@ operator matches dense matrix multiply."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert jnp.allclose(simple_sparse @ x, simple_dense @ x)


def test_matmul_operator_jit(simple_sparse):
    """JIT-compiled @ operator matches eager @ result."""
    x = jnp.ones(simple_sparse.shape[1])
    assert jnp.allclose(jax.jit(lambda sp, v: sp @ v)(simple_sparse, x), simple_sparse @ x)


# ---------- __getitem__ ----------


def test_getitem_existing_entry(simple_dense, simple_sparse):
    """__getitem__ returns the correct value for stored entries."""
    assert jnp.allclose(simple_sparse[0, 0], simple_dense[0, 0])
    assert jnp.allclose(simple_sparse[0, 2], simple_dense[0, 2])
    assert jnp.allclose(simple_sparse[1, 3], simple_dense[1, 3])
    assert jnp.allclose(simple_sparse[2, 1], simple_dense[2, 1])


def test_getitem_zero_entry(simple_sparse):
    """__getitem__ returns 0.0 for structurally-zero entries."""
    assert jnp.allclose(simple_sparse[0, 1], 0.0)
    assert jnp.allclose(simple_sparse[1, 0], 0.0)


def test_getitem_returns_zero_dtype(simple_sparse):
    """Zero returned for missing entries has the matrix dtype."""
    val = simple_sparse[0, 1]
    assert val.dtype == simple_sparse.dtype


def test_getitem_jit_compatible(simple_sparse):
    """__getitem__ works correctly inside jax.jit."""
    f = jax.jit(lambda sp: sp[2, 1])
    assert jnp.allclose(f(simple_sparse), simple_sparse[2, 1])


def test_getitem_out_of_range_returns_zero(simple_sparse):
    """__getitem__ returns 0.0 for an absent (but in-bounds) entry."""
    # (2, 3) is a valid index but the entry is 5.0, not zero — test a truly absent one
    assert jnp.allclose(simple_sparse[0, 3], 0.0)
