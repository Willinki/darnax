"""Detailed comparison: dense matmul vs block-sparse matmul.

Parameter sweep focused on the sparse regime (density 0.001–0.1)
and realistic sizes (n 128–1024), with batch size 16.

Run with:
    uv run pytest bench/bench_sparse_vs_dense.py --benchmark-only \
        --benchmark-group-by=param:n,param:density \
        --benchmark-columns=mean,stddev,rounds \
        --benchmark-histogram=bench/histograms/sparse_vs_dense \
        --no-cov -v

Histograms are written to bench/histograms/ as SVG files (one per group).
Omit --benchmark-histogram if you don't need the plots.
"""

from __future__ import annotations

import warnings

import jax
import pytest

from darnax.utils.block_sparse_matmul import block_sparse_matvec, make_block_sparse
from darnax.utils.sparse_matmul import to_sparse

# ---------- Parameter grid ----------

NS = [512, 1024, 2048, 4096]
DENSITIES = [0.001, 0.005, 0.01, 0.05]
BLOCK_SIZES = [16, 32]
BATCH_SIZE = 16


# ---------- Helpers ----------


def _make_inputs(n: int, density: float, seed: int = 0):
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    dense = jax.random.normal(k1, (n, n))
    mask = jax.random.bernoulli(k2, density, (n, n))
    dense = (dense * mask).block_until_ready()
    sp = to_sparse(dense)
    x = jax.random.normal(jax.random.key(seed + 1), (n, BATCH_SIZE)).block_until_ready()
    return dense, sp, x


def _make_block_sparse_inputs(n: int, density: float, b: int, seed: int = 0):
    k_row, k_col = n // b, n // b
    n_total = k_row * k_col
    n_active = max(1, min(n_total, round(density * n_total)))
    x = jax.random.normal(jax.random.key(seed + 1), (n, BATCH_SIZE)).block_until_ready()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mat = make_block_sparse(jax.random.key(seed), n, n, b, n_active)
    return mat, x


_jit_dense = jax.jit(lambda w, x: w @ x)

# Warm up dense kernel at module load.
_d, _, _x = _make_inputs(128, 0.05)
_jit_dense(_d, _x).block_until_ready()


# ---------- Benchmarks ----------


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("density", DENSITIES)
def test_dense(benchmark, n, density):
    """Benchmark dense matrix–vector multiply."""
    dense, _, x = _make_inputs(n, density)
    _jit_dense(dense, x).block_until_ready()  # ensure compiled for this shape
    benchmark(lambda: _jit_dense(dense, x).block_until_ready())


@pytest.mark.parametrize("b", BLOCK_SIZES)
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("density", DENSITIES)
def test_block_sparse(benchmark, n, density, b):
    """Benchmark block-sparse matrix–vector multiply."""
    if n % b != 0:
        pytest.skip(f"n={n} not divisible by b={b}")
    mat, x = _make_block_sparse_inputs(n, density, b)
    jit_fn = jax.jit(block_sparse_matvec)
    jit_fn(mat, x).block_until_ready()  # compile
    benchmark(lambda: jit_fn(mat, x).block_until_ready())
