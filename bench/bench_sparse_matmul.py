"""Benchmark: dense matmul vs COO sparse_matvec vs block_sparse_matvec.

Run with:
    pytest bench/bench_sparse_matmul.py -v --benchmark-group-by=param:n,param:density --no-cov

Each benchmark is JIT-compiled once before timing; `block_until_ready` ensures
we measure actual compute and not just dispatch.

Note on JIT scope
-----------------
The sparse index arrays are stored as static aux in both SparseMatrix and
BlockSparseMatrix pytrees, so JAX recompiles for each distinct sparsity pattern.
Each test creates its own jit instance to avoid aux cache collisions.
Dense matmul has no such constraint and shares one compiled fn.

Block-sparse parameters
-----------------------
BLOCK_SIZES controls the b×b block granularity candidates.  Only (n, b) pairs
where n % b == 0 are benchmarked; others are skipped.  density maps to
n_active_blocks via:
    n_active = max(1, round(density * k_row * k_col))
"""

from __future__ import annotations

import warnings

import jax
import pytest

from darnax.utils.block_sparse_matmul import block_sparse_matvec, make_block_sparse
from darnax.utils.sparse_matmul import sparse_matvec, to_sparse

# ---------- Parameter grid ----------

NS = [32, 64, 128, 256]
DENSITIES = [0.01, 0.1, 0.5, 0.9, 0.99]  # fraction of NON-zero entries
BLOCK_SIZES = [2, 4, 8, 16]  # b for block-sparse; skipped when n % b != 0


# ---------- Helpers ----------


def _make_inputs(n: int, density: float, seed: int = 0):
    """Return a (dense, coo_sparse, x) triple."""
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    dense = jax.random.normal(k1, (n, n))
    mask = jax.random.bernoulli(k2, density, (n, n))
    dense = (dense * mask).block_until_ready()
    sp = to_sparse(dense)
    x = jax.random.normal(jax.random.key(seed + 1), (n,)).block_until_ready()
    return dense, sp, x


def _make_block_sparse_inputs(n: int, density: float, b: int, seed: int = 0):
    """Return a (block_sparse_mat, x) pair at approximately the requested density."""
    k_row = n // b
    k_col = n // b
    n_total = k_row * k_col
    n_active = max(1, min(n_total, round(density * n_total)))
    x = jax.random.normal(jax.random.key(seed + 1), (n,)).block_until_ready()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mat = make_block_sparse(jax.random.key(seed), n, n, b, n_active)
    return mat, x


# Dense matmul is pattern-agnostic: one shared JIT is fine.
_jit_dense = jax.jit(lambda w, x: w @ x)


def _run_dense(w, x):
    return _jit_dense(w, x).block_until_ready()


# Warm up dense kernel at module load.
_d, _, _x = _make_inputs(32, 0.5)
_run_dense(_d, _x)


# ---------- Benchmarks ----------


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("density", DENSITIES)
def test_dense_matmul(benchmark, n, density):
    """Benchmark dense matmul for a given (n, density) configuration."""
    dense, _, x = _make_inputs(n, density)
    _run_dense(dense, x)  # ensure compiled for this shape
    benchmark(_run_dense, dense, x)


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("density", DENSITIES)
def test_coo_sparse_matmul(benchmark, n, density):
    """Benchmark COO sparse_matvec for a given (n, density) configuration."""
    _, sp, x = _make_inputs(n, density)
    jit_fn = jax.jit(sparse_matvec)

    def run(sp, x):
        return jit_fn(sp, x).block_until_ready()

    run(sp, x)  # compile
    benchmark(run, sp, x)


@pytest.mark.parametrize("b", BLOCK_SIZES)
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("density", DENSITIES)
def test_block_sparse_matmul(benchmark, n, density, b):
    """Benchmark block_sparse_matvec for a given (n, density, b) configuration."""
    if n % b != 0:
        pytest.skip(f"n={n} not divisible by b={b}")
    mat, x = _make_block_sparse_inputs(n, density, b)
    jit_fn = jax.jit(block_sparse_matvec)

    def run(mat, x):
        return jit_fn(mat, x).block_until_ready()

    run(mat, x)  # compile
    benchmark(run, mat, x)
