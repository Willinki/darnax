"""Block-sparse matmul with a Pallas kernel.

Replaces the JAX-level ``segment_sum`` scatter in
:mod:`darnax.block_sparse_matmul` with a Pallas kernel that issues one GPU
thread-block per active sparse block and scatters contributions via
``atomic_add``.

Kernel design
-------------
Grid ``(n_active,)``.  Program ``i`` handles active block ``i``:

1. Reads ``values[i]`` — (b, b) weight block.
2. Reads ``x_slices[i]`` — (b,) pre-gathered input slice.
3. Computes ``contrib = values[i] @ x_slices[i]`` — (b,) partial output.
4. Atomically adds each ``contrib[k]`` into ``y[row_start_i + k]``.

The x-slice pre-gather is a regular JAX gather executed before the kernel
(read-only, coalesced after blocks are sorted by column in
:func:`~darnax.block_sparse_matmul.make_block_sparse`).

Platform notes
--------------
* **CUDA / ROCm GPU**: runs natively via the Triton Pallas backend.
* **TPU**: not tested; use the Mosaic backend.
* **CPU / Apple Metal**: pass ``interpret=True`` to :func:`block_sparse_matvec_pallas`
  (or set the module-level ``_INTERPRET`` flag).  The Pallas interpreter runs the
  kernel sequentially — correct but without GPU-level throughput.

Re-exports
----------
:class:`~darnax.block_sparse_matmul.BlockSparseMatrix` and
:func:`~darnax.block_sparse_matmul.make_block_sparse` are re-exported for
convenience so callers can use this module as a drop-in replacement.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp

from darnax.utils.block_sparse_matmul import (  # noqa: F401 — re-export
    BlockSparseMatrix,
    make_block_sparse,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

__all__ = [
    "block_sparse_matvec_pallas",
    "BlockSparseMatrix",
    "make_block_sparse",
]

# ---------------------------------------------------------------------------
# Auto-detect interpreter mode: use interpret=True on CPU / Metal so that
# the kernel runs without requiring a CUDA/ROCm device.  Override by setting
# the env variable BSM_PALLAS_INTERPRET=0 to force native compilation.
# ---------------------------------------------------------------------------
_backend = jax.default_backend()
_INTERPRET: bool = _backend not in ("gpu", "tpu")
if os.environ.get("BSM_PALLAS_INTERPRET", "").strip() == "0":
    _INTERPRET = False


# ---------------------------------------------------------------------------
# Pallas kernel
# ---------------------------------------------------------------------------


def _make_kernel(b: int) -> Callable[..., None]:
    """Return a Pallas kernel closure for a given block size ``b``.

    Each program in the ``(n_active,)`` grid handles one active block:

    - ``values_ref``   — shape ``(1, b, b)`` (one block matrix, batched dim first)
    - ``x_slices_ref`` — shape ``(1, b)``    (pre-gathered x slice)
    - ``row_starts_ref`` — shape ``(1,)``    (int32 start-row of this block in y)
    - ``_y_in_ref``    — shape ``(m,)``      (aliased input, same buffer as y_out)
    - ``y_out_ref``    — shape ``(m,)``      (output accumulator, atomic scatter target)
    """

    def kernel(
        values_ref: Any,
        x_slices_ref: Any,
        row_starts_ref: Any,
        _y_in_ref: Any,
        y_out_ref: Any,
    ) -> None:
        # Pallas calls kernel(*in_refs, *out_refs).  With input_output_aliases
        # {3: 0}, input 3 (_y_in_ref) and output 0 (y_out_ref) are the same
        # underlying buffer; we only need to write through y_out_ref.
        vals = values_ref[0]  # (b, b)
        x_s = x_slices_ref[0]  # (b,)
        contrib = vals @ x_s  # (b,)
        row_start = row_starts_ref[0]  # scalar int32

        # Unrolled atomic scatter: each iteration writes one scalar element.
        # Python for-loop is unrolled at trace time → b separate atomic ops.
        for k in range(b):
            pl.atomic_add(y_out_ref, (row_start + k,), contrib[k])

    return kernel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def block_sparse_matvec_pallas(
    mat: BlockSparseMatrix,
    x: Array,
    *,
    interpret: bool | None = None,
) -> Array:
    """Block-sparse matrix–vector product via a Pallas kernel.

    Parameters
    ----------
    mat : BlockSparseMatrix
        Block-sparse matrix with logical shape ``(m, n)``.
    x : Array, shape ``(n,)`` or ``(batch, n)``
        Dense input vector or batch of vectors.  For 2-D input the batch
        dimension is mapped via ``jax.vmap``.
    interpret : bool or None
        If ``True`` run in the Pallas interpreter (CPU/Metal safe, sequential).
        If ``False`` compile natively (requires CUDA/ROCm).
        ``None`` (default) uses the module-level auto-detected ``_INTERPRET`` flag.

    Returns
    -------
    Array, shape ``(m,)`` or ``(batch, m)``

    """
    if interpret is None:
        interpret = _INTERPRET

    if x.ndim == 2:
        return jax.vmap(lambda v: block_sparse_matvec_pallas(mat, v, interpret=interpret))(x)

    m, _n = mat.shape
    b = mat.b
    n_active = mat.n_active

    # --- Step 1: pre-gather x slices (regular JAX, coalesced) ---------------
    col_starts = mat.block_col_idx * b  # (n_active,)
    x_slices = x[col_starts[:, None] + jnp.arange(b)]  # (n_active, b)

    row_starts = (mat.block_row_idx * b).astype(jnp.int32)  # (n_active,)
    y_init = jnp.zeros(m, dtype=x.dtype)

    # --- Step 2: Pallas scatter kernel --------------------------------------
    # y_full_spec: every program sees the whole (m,) buffer so that
    # atomic_add with a dynamic row_start hits the correct address.
    # JAX 0.4.38 BlockSpec signature: BlockSpec(block_shape, index_map).
    y_full_spec = pl.BlockSpec((m,), lambda _: (0,))

    return cast(
        "Array",
        pl.pallas_call(
            _make_kernel(b),
            out_shape=jax.ShapeDtypeStruct((m,), x.dtype),  # type: ignore[no-untyped-call]
            grid=(n_active,),
            in_specs=[
                pl.BlockSpec((1, b, b), lambda i: (i, 0, 0)),  # values[i]
                pl.BlockSpec((1, b), lambda i: (i, 0)),  # x_slices[i]
                pl.BlockSpec((1,), lambda i: (i,)),  # row_starts[i]
                y_full_spec,  # y_init (aliased output)
            ],
            # Give every program a view of the full output so atomic indexing
            # with a dynamic row_start works correctly.
            out_specs=y_full_spec,
            # y_init is both input (positional index 3) and the sole output.
            input_output_aliases={3: 0},
            interpret=interpret,
        )(mat.values, x_slices, row_starts, y_init),
    )
