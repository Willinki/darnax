r"""Benchmark: dense batched matmul vs BCOO sparse batched matmul.

Compares ``W @ X`` where W is (n, n) and X is (n, batch_size) across a range
of matrix sizes and densities.

BCOO is JAX's native compressed sparse format (``jax.experimental.sparse``).
Its ``@`` operator dispatches to ``bcoo_dot_general`` which, on GPU, can use
cuSPARSE kernels; on CPU/Metal it falls back to a gather-scatter implementation.

Run with:
    uv run pytest bench/bench_bcoo_vs_dense.py --benchmark-only \\
        --benchmark-group-by=param:n,param:density \\
        --benchmark-columns=mean,stddev,rounds \\
        --benchmark-histogram=bench/histograms/bcoo_vs_dense \\
        --no-cov -v

Note on JIT and BCOO structure
-------------------------------
BCOO indices are stored as a *dynamic* array (not static aux), so a single
jit-compiled function works across different sparsity patterns of the same shape.
This is unlike the custom COO in sparse_matmul.py which bakes indices into aux
and forces a recompile per pattern.
"""

from __future__ import annotations

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import pytest

# ---------- Parameter grid ----------

NS = [512, 1024, 2048, 4096]
DENSITIES = [0.1, 0.05, 0.01, 0.005, 0.001]
BATCH_SIZE = 16


# ---------- Helpers ----------


def _make_inputs(n: int, density: float, seed: int = 0):
    """Return a (dense_w, bcoo_w, x) triple for benchmarking.

    ``dense_w`` is the full (n, n) weight matrix with approximately
    ``density`` fraction of non-zero entries.  ``bcoo_w`` is the BCOO
    conversion of the same matrix.  ``x`` is a (n, BATCH_SIZE) input.
    """
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    w = jax.random.normal(k1, (n, n))
    mask = jax.random.bernoulli(k2, density, (n, n))
    w_sparse = (w * mask).block_until_ready()
    bcoo_w = jsparse.BCOO.fromdense(w_sparse)
    x = jax.random.normal(jax.random.key(seed + 1), (n, BATCH_SIZE)).block_until_ready()
    return w_sparse, bcoo_w, x


# One shared JIT for dense (pattern-agnostic).
_jit_dense = jax.jit(jnp.matmul)

# One shared JIT for BCOO (indices are dynamic — no recompile per pattern).
_jit_bcoo = jax.jit(jsparse.BCOO.todense)  # used only for warm-up shape check
_jit_bcoo_matmul = jax.jit(lambda w, x: w @ x)

# Warm up both kernels at module load so the first benchmark isn't penalised.
_w, _bw, _x = _make_inputs(512, 0.05)
_jit_dense(_w, _x).block_until_ready()
_jit_bcoo_matmul(_bw, _x).block_until_ready()


# ---------- Benchmarks ----------


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("density", DENSITIES)
def test_dense_batched_matmul(benchmark, n, density):
    """Benchmark dense (n, n) @ (n, batch) multiply."""
    w, _, x = _make_inputs(n, density)
    _jit_dense(w, x).block_until_ready()  # ensure compiled for this shape
    benchmark(lambda: _jit_dense(w, x).block_until_ready())


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("density", DENSITIES)
def test_bcoo_batched_matmul(benchmark, n, density):
    """Benchmark BCOO sparse (n, n) @ (n, batch) multiply."""
    _, bcoo_w, x = _make_inputs(n, density)
    _jit_bcoo_matmul(bcoo_w, x).block_until_ready()  # ensure compiled for this shape
    benchmark(lambda: _jit_bcoo_matmul(bcoo_w, x).block_until_ready())
