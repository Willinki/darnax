"""Tests for block_sparse_matmul_pallas (Pallas-kernel implementation).

All tests run with ``interpret=True`` so they work on CPU / Apple Metal without
a CUDA device.  The correctness baseline is the reference JAX implementation
:func:`~darnax.block_sparse_matmul.block_sparse_matvec`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from darnax.block_sparse_matmul import block_sparse_matvec, make_block_sparse
from darnax.block_sparse_matmul_pallas import block_sparse_matvec_pallas

KEY = jax.random.key(0)

# Force interpreter mode for all tests (portable: CPU / Metal / GPU).
_INTERP = {"interpret": True}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mat():
    """16×16 block-sparse matrix, b=4, 12 active blocks."""
    return make_block_sparse(KEY, m=16, n=16, b=4, n_active_blocks=12)


# ---------------------------------------------------------------------------
# Correctness vs. reference implementation
# ---------------------------------------------------------------------------


def test_pallas_matches_reference(mat):
    """Pallas kernel output matches the reference JAX implementation."""
    x = jax.random.normal(jax.random.key(1), (mat.shape[1],))
    ref = block_sparse_matvec(mat, x)
    got = block_sparse_matvec_pallas(mat, x, **_INTERP)
    assert jnp.allclose(ref, got, atol=1e-5), f"max diff: {jnp.max(jnp.abs(ref - got))}"


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_pallas_matches_reference_random(seed):
    """Pallas kernel matches reference for random seeds."""
    key = jax.random.key(seed)
    m, n, b = 16, 12, 4
    mat = make_block_sparse(key, m=m, n=n, b=b, n_active_blocks=6)
    x = jax.random.normal(jax.random.key(seed + 100), (n,))
    ref = block_sparse_matvec(mat, x)
    got = block_sparse_matvec_pallas(mat, x, **_INTERP)
    assert jnp.allclose(ref, got, atol=1e-5)


def test_pallas_matches_dense():
    """Pallas output matches the equivalent dense matmul."""
    key = jax.random.key(2)
    m, n, b = 16, 16, 4
    mat = make_block_sparse(key, m=m, n=n, b=b, n_active_blocks=12)

    dense = jnp.zeros((m, n))
    for k in range(mat.n_active):
        br = int(mat.block_row_idx[k])
        bc = int(mat.block_col_idx[k])
        dense = dense.at[br * b : (br + 1) * b, bc * b : (bc + 1) * b].set(mat.values[k])

    x = jax.random.normal(jax.random.key(3), (n,))
    got = block_sparse_matvec_pallas(mat, x, **_INTERP)
    assert jnp.allclose(got, dense @ x, atol=1e-5)


# ---------------------------------------------------------------------------
# Shape / dtype
# ---------------------------------------------------------------------------


def test_output_shape(mat):
    """Output shape is (m,) for a (m, n) matrix and (n,) input."""
    x = jax.random.normal(KEY, (mat.shape[1],))
    y = block_sparse_matvec_pallas(mat, x, **_INTERP)
    assert y.shape == (mat.shape[0],)


def test_output_dtype_float32(mat):
    """Output dtype matches float32 input dtype."""
    x = jax.random.normal(KEY, (mat.shape[1],))
    y = block_sparse_matvec_pallas(mat, x, **_INTERP)
    assert y.dtype == jnp.float32


def test_non_square():
    """Pallas kernel works for non-square matrices."""
    key = jax.random.key(3)
    mat = make_block_sparse(key, m=12, n=8, b=4, n_active_blocks=6)
    x = jax.random.normal(key, (8,))
    y = block_sparse_matvec_pallas(mat, x, **_INTERP)
    assert y.shape == (12,)


# ---------------------------------------------------------------------------
# Block sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("b", [4, 8, 16])
def test_various_block_sizes(b):
    """Pallas kernel matches reference across different block sizes."""
    key = jax.random.key(0)
    m, n = 4 * b, 4 * b
    mat = make_block_sparse(key, m=m, n=n, b=b, n_active_blocks=8)
    x = jax.random.normal(jax.random.key(1), (n,))
    ref = block_sparse_matvec(mat, x)
    got = block_sparse_matvec_pallas(mat, x, **_INTERP)
    assert jnp.allclose(ref, got, atol=1e-5)


# ---------------------------------------------------------------------------
# JIT
# ---------------------------------------------------------------------------


def test_jit(mat):
    """Pallas kernel produces the same result under jit."""
    x = jax.random.normal(KEY, (mat.shape[1],))
    eager = block_sparse_matvec_pallas(mat, x, **_INTERP)
    jitted = jax.jit(lambda m, v: block_sparse_matvec_pallas(m, v, **_INTERP))(mat, x)
    assert jnp.allclose(eager, jitted, atol=1e-5)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_grad_through_values(mat):
    """Gradients flow through BlockSparseMatrix.values in the Pallas kernel."""
    x = jax.random.normal(KEY, (mat.shape[1],))

    def f(m):
        return block_sparse_matvec_pallas(m, x, **_INTERP).sum()

    grads = jax.grad(f)(mat)
    assert grads.values.shape == mat.values.shape
    assert jnp.all(jnp.isfinite(grads.values))


def test_grad_matches_reference(mat):
    """Gradients from Pallas kernel match reference implementation."""
    x = jax.random.normal(KEY, (mat.shape[1],))

    def f_ref(m):
        return block_sparse_matvec(m, x).sum()

    def f_pal(m):
        return block_sparse_matvec_pallas(m, x, **_INTERP).sum()

    g_ref = jax.grad(f_ref)(mat)
    g_pal = jax.grad(f_pal)(mat)
    assert jnp.allclose(g_ref.values, g_pal.values, atol=1e-5)
