"""Block-sparse recurrent matrix for JAX.

A block-sparse matrix ``J`` of shape ``(m, n)`` is stored as a fixed set of
``n_active`` dense blocks of size ``b×b``, placed at randomly sampled positions
in the ``(k_row, k_col)`` block grid where ``k_row = m/b`` and
``k_col = n/b``.  ``m`` and ``n`` must be exactly divisible by ``b``.
All other entries are exactly zero and never stored.

Block positions are fixed at construction time and treated as static aux data
in the JAX PyTree, so they are baked into the XLA binary as constants.  Only
``values`` (shape ``(n_active, b, b)``) is a trainable dynamic leaf.

Matrix-vector product y = J @ x is implemented via:
  1. static gathers of ``b``-element slices of ``x`` for each active column block
  2. batched block matmul  ``(n_active, b, b) @ (n_active, b) -> (n_active, b)``
  3. ``segment_sum`` scatter into output rows

This avoids random per-element index gathers (cache-hostile on GPU) in favour
of contiguous block accesses with a single scatter at the end.

Giant-component guarantee
-------------------------
The block meta-graph is Erdős–Rényi on ``k_row`` nodes.  The average number of
active blocks per row must exceed ``er_safety_factor`` (default 2.0) to ensure
connectivity::

    avg_blocks_per_row = n_active / k_row > er_safety_factor

:func:`make_block_sparse` warns when this condition is violated but does not
raise, so callers can intentionally use sparser graphs.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import DTypeLike

    KeyArray = jax.Array


@register_pytree_node_class
@dataclass(frozen=True)
class BlockSparseMatrix:
    """Block-sparse matrix as a JAX PyTree.

    Only ``values`` is a dynamic leaf; all other fields are static aux data
    baked into the XLA binary as constants.  JIT retraces only on shape change.

    Parameters
    ----------
    values : Array, shape ``(n_active, b, b)``
        Dense values for each active block.
    block_row_idx : Array, shape ``(n_active,)``, int
        Block-row index (in ``[0, k_row)``) for each active block.
    block_col_idx : Array, shape ``(n_active,)``, int
        Block-column index (in ``[0, k_col)``) for each active block.
    shape : tuple[int, int]
        ``(m, n)`` — logical shape of the represented matrix.
    b : int
        Block side length.
    dtype : jnp.dtype
        dtype of ``values``.

    Notes
    -----
    ``m`` and ``n`` must be exactly divisible by ``b``.
    ``k_row = m // b``, ``k_col = n // b``.

    """

    values: Array
    block_row_idx: Array
    block_col_idx: Array
    shape: tuple[int, int]
    b: int
    dtype: jnp.dtype

    # ---------- PyTree protocol ----------

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[Array],
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, int], int, jnp.dtype],
    ]:
        """Flatten into ``(values,)`` leaf and static index/shape aux."""
        children = (self.values,)
        aux = (
            tuple(np.asarray(self.block_row_idx).tolist()),
            tuple(np.asarray(self.block_col_idx).tolist()),
            self.shape,
            self.b,
            self.dtype,
        )
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[tuple[int, ...], tuple[int, ...], tuple[int, int], int, jnp.dtype],
        children: tuple[Array],
    ) -> BlockSparseMatrix:
        """Reconstruct a :class:`BlockSparseMatrix` from aux data and leaf."""
        row_tup, col_tup, shape, b, dtype = aux
        (values,) = children
        return cls(
            values=values,
            block_row_idx=jnp.array(row_tup, dtype=jnp.int32),
            block_col_idx=jnp.array(col_tup, dtype=jnp.int32),
            shape=shape,
            b=b,
            dtype=dtype,
        )

    # ---------- Convenience ----------

    @property
    def n_active(self) -> int:
        """Number of active (stored) blocks."""
        return int(self.values.shape[0])

    def __matmul__(self, x: Array) -> Array:
        """Compute ``self @ x`` via :func:`block_sparse_matvec`."""
        return block_sparse_matvec(self, x)


# ---------- Matrix-vector multiply ----------


def block_sparse_matvec(mat: BlockSparseMatrix, x: Array) -> Array:
    """Block-sparse matrix–vector product ``y = mat @ x``.

    Parameters
    ----------
    mat : BlockSparseMatrix
        Block-sparse matrix with logical shape ``(m, n)``.
    x : Array, shape ``(n,)`` or ``(batch, n)``
        Dense input vector or batch of vectors.  When ``x`` is 2-D the result
        has shape ``(batch, m)`` and the batch is mapped via ``jax.vmap``.

    Returns
    -------
    Array, shape ``(m,)`` or ``(batch, m)``
        Result ``y = mat @ x``.

    Notes
    -----
    Implementation outline:

    1. Gather a ``(n_active, b)`` slice tensor via static block-column indices
       (baked as XLA constants).  Blocks are stored sorted by column so the
       gather accesses ``x`` in roughly ascending order.
    2. Batched block matmul: ``(n_active, b, b) @ (n_active, b, 1) ->
       (n_active, b)`` via cuBLAS batched GEMM.
    3. ``segment_sum`` scatter of the flattened ``(n_active * b,)`` contributions
       into ``m`` output slots.

    """
    if x.ndim == 2:
        return jax.vmap(lambda v: block_sparse_matvec(mat, v))(x)

    m, _ = mat.shape
    b = mat.b

    # Step 1: gather x slices — shape (n_active, b)
    col_starts = mat.block_col_idx * b  # (n_active,) — sorted ascending
    x_slices = x[col_starts[:, None] + jnp.arange(b)]  # (n_active, b)

    # Step 2: batched block matmul — (n_active, b, b) @ (n_active, b, 1) -> (n_active, b)
    contribs = (mat.values @ x_slices[..., None]).squeeze(-1)  # (n_active, b)

    # Step 3: scatter via segment_sum
    row_starts = mat.block_row_idx * b  # (n_active,)
    row_idx_flat = (row_starts[:, None] + jnp.arange(b)).reshape(-1)  # (n_active * b,)
    contribs_flat = contribs.reshape(-1)  # (n_active * b,)

    return jax.ops.segment_sum(contribs_flat, row_idx_flat, num_segments=m)


# ---------- Factory ----------


def make_block_sparse(
    key: KeyArray,
    m: int,
    n: int,
    b: int,
    n_active_blocks: int,
    *,
    er_safety_factor: float = 2.0,
    strength: float = 1.0,
    dtype: DTypeLike = jnp.float32,
) -> BlockSparseMatrix:
    """Construct a randomly initialised :class:`BlockSparseMatrix`.

    Block positions are sampled uniformly without replacement from the full
    ``(k_row, k_col)`` block grid.  Values are initialised i.i.d. Gaussian
    scaled to unit variance at each output neuron given the average fan-in.

    Parameters
    ----------
    key : KeyArray
        JAX PRNG key.  Split internally; the same key can be reused by the
        caller.
    m : int
        Number of rows in the logical matrix.
    n : int
        Number of columns in the logical matrix.
    b : int
        Block side length.  Both ``m`` and ``n`` must be divisible by ``b``.
    n_active_blocks : int
        Number of non-zero blocks to place.  Must be ``<= k_row * k_col``.
    er_safety_factor : float, default 2.0
        Warn if ``n_active_blocks / k_row <= er_safety_factor`` (Erdős–Rényi
        giant-component threshold).
    strength : float, default 1.0
        Scalar multiplied into the Gaussian initialisation, analogous to the
        ``strength`` parameter in dense recurrent modules.
    dtype : DTypeLike, default ``jnp.float32``
        dtype for ``values``.

    Returns
    -------
    BlockSparseMatrix

    Raises
    ------
    ValueError
        If ``m`` or ``n`` is not divisible by ``b``, or if
        ``n_active_blocks > k_row * k_col``.

    Warns
    -----
    UserWarning
        If the Erdős–Rényi giant-component condition is not met.

    """
    if m % b != 0:
        raise ValueError(f"m={m} is not divisible by b={b}")
    if n % b != 0:
        raise ValueError(f"n={n} is not divisible by b={b}")

    k_row = m // b
    k_col = n // b
    n_total = k_row * k_col

    if n_active_blocks > n_total:
        raise ValueError(
            f"n_active_blocks={n_active_blocks} exceeds total block slots "
            f"k_row*k_col={k_row}*{k_col}={n_total}"
        )

    avg_blocks_per_row = n_active_blocks / k_row
    if avg_blocks_per_row <= er_safety_factor:
        warnings.warn(
            f"avg_blocks_per_row={avg_blocks_per_row:.3f} <= "
            f"er_safety_factor={er_safety_factor:.3f}. "
            "Giant component of the block meta-graph is not guaranteed; "
            "the network may be disconnected.",
            UserWarning,
            stacklevel=2,
        )

    key_pos, key_val = jax.random.split(key)

    # Sample block positions without replacement
    flat_idx = jax.random.choice(key_pos, n_total, shape=(n_active_blocks,), replace=False)
    block_row_idx = (flat_idx // k_col).astype(jnp.int32)
    block_col_idx = (flat_idx % k_col).astype(jnp.int32)

    # Sort by column so the gather in block_sparse_matvec accesses x in
    # ascending order — improves cache / memory-coalescing on GPU.
    sort_order = jnp.argsort(block_col_idx, stable=True)
    block_row_idx = block_row_idx[sort_order]
    block_col_idx = block_col_idx[sort_order]

    # Gaussian init scaled to unit output variance:
    # avg fan-in per output neuron = (n_active_blocks / k_row) * b
    avg_fan_in = (n_active_blocks / k_row) * b
    scale = strength / math.sqrt(avg_fan_in)
    values = jax.random.normal(key_val, shape=(n_active_blocks, b, b), dtype=dtype) * scale

    return BlockSparseMatrix(
        values=values,
        block_row_idx=block_row_idx,
        block_col_idx=block_col_idx,
        shape=(m, n),
        b=b,
        dtype=jnp.dtype(dtype),
    )
