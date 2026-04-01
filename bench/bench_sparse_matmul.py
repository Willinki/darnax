"""Benchmark: sparse_matvec vs dense matmul across sparsity levels and matrix sizes.

Run with:
    pytest bench/bench_sparse_matmul.py -v --benchmark-group-by=param:n,param:sparsity --no-cov

Each benchmark is JIT-compiled once before timing; `block_until_ready` ensures
we measure actual compute and not just dispatch.

Note on JIT scope
-----------------
The sparse index arrays (row_indices, col_indices, linear_indices) are stored as
static aux in the SparseMatrix pytree, so JAX recompiles for each distinct sparsity
pattern.  Using a shared module-level jit across different patterns would cause a
cache-lookup failure (shape mismatch on aux comparison), so each test creates its
own jit instance.  Dense matmul has no such constraint and shares one compiled fn.
"""

from __future__ import annotations

import jax
import pytest

from darnax.utils.sparse_matmul import sparse_matvec, to_sparse

# ---------- Parameter grid ----------

NS = [32, 64, 128, 256]
SPARSITIES = [0.1, 0.5, 0.9, 0.99, 0.01]  # fraction of NON-zero entries


# ---------- Helpers ----------


def _make_inputs(n: int, sparsity: float, seed: int = 0):
    """Return a (dense, sparse, x) triple."""
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    dense = jax.random.normal(k1, (n, n))
    mask = jax.random.bernoulli(k2, sparsity, (n, n))
    dense = (dense * mask).block_until_ready()
    sp = to_sparse(dense)
    x = jax.random.normal(jax.random.key(seed + 1), (n,)).block_until_ready()
    return dense, sp, x


# Dense matmul is pattern-agnostic: one shared JIT is fine.
_jit_dense = jax.jit(lambda w, x: w @ x)


def _run_dense(w, x):
    return _jit_dense(w, x).block_until_ready()


# Warm up the dense kernel at module load.
_d, _, _x = _make_inputs(32, 0.5)
_run_dense(_d, _x)


# ---------- Benchmarks ----------


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("sparsity", SPARSITIES)
def test_dense_matmul(benchmark, n, sparsity):
    """Benchmark dense matmul for a given (n, sparsity) configuration."""
    dense, _, x = _make_inputs(n, sparsity)
    _run_dense(dense, x)  # ensure compiled for this shape
    benchmark(_run_dense, dense, x)


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("sparsity", SPARSITIES)
def test_sparse_matmul(benchmark, n, sparsity):
    """Benchmark sparse_matvec for a given (n, sparsity) configuration."""
    _, sp, x = _make_inputs(n, sparsity)
    # Each test gets its own jit so aux comparison never crosses sparsity patterns.
    jit_fn = jax.jit(sparse_matvec)

    def run(sp, x):
        return jit_fn(sp, x).block_until_ready()

    run(sp, x)  # compile
    benchmark(run, sp, x)
