"""Sparse matrix representation and matrix-vector multiply for JAX.

A sparse matrix is stored as a triplet ``(row_indices, col_indices, values)``
where only the non-zero entries are kept.  The triplet is registered as a JAX
PyTree so that JAX transforms (``jit``, ``grad``, ``vmap``, …) see ``values``
as the single dynamic leaf while ``row_indices``, ``col_indices``, and the
shape are static aux data.

Intentionally avoids ``jax.experimental.sparse`` / BCOO so the representation
stays minimal and transparent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

if TYPE_CHECKING:
    from jax import Array


@register_pytree_node_class
@dataclass(frozen=True)
class SparseMatrix:
    """COO-style sparse matrix as a JAX PyTree.

    Parameters
    ----------
    row_indices : Array, shape (nnz,), int
        Row index of each non-zero entry.
    col_indices : Array, shape (nnz,), int
        Column index of each non-zero entry.
    values : Array, shape (nnz,)
        Non-zero values.
    shape : tuple[int, int]
        ``(n_rows, n_cols)`` of the dense matrix this represents.
    dtype : jnp.dtype
        dtype of ``values``.
    linear_indices : Array, shape (nnz,), int
        ``row * n_cols + col`` for each entry, sorted ascending.  Used for
        O(log nnz) lookup in :meth:`__getitem__`.

    Notes
    -----
    Only ``values`` is a dynamic leaf; ``row_indices``, ``col_indices``,
    ``shape``, and ``dtype`` are static aux data.  This means JIT recompiles when sparsity
    pattern changes, but gradients flow through ``values`` correctly.

    """

    row_indices: Array
    col_indices: Array
    values: Array
    shape: tuple[int, int]
    dtype: jnp.dtype
    linear_indices: Array  # row * n_cols + col, sorted ascending

    # ---------- PyTree protocol ----------

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[Array],
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, int], jnp.dtype],
    ]:
        """Flatten into (values,) leaf and static index/shape aux."""
        children = (self.values,)
        # Index arrays must be hashable/comparable for JIT cache lookup, so store
        # as Python tuples.  The conversion is a one-time cost at trace time.
        import numpy as np

        aux = (
            tuple(np.asarray(self.row_indices).tolist()),
            tuple(np.asarray(self.col_indices).tolist()),
            tuple(np.asarray(self.linear_indices).tolist()),
            self.shape,
            self.dtype,
        )
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, int], jnp.dtype],
        children: tuple[Array],
    ) -> SparseMatrix:
        """Reconstruct a SparseMatrix from aux data and (values,) leaf."""
        row_tup, col_tup, lin_tup, shape, dtype = aux
        (values,) = children
        return cls(
            jnp.array(row_tup, dtype=jnp.int32),
            jnp.array(col_tup, dtype=jnp.int32),
            values,
            shape,
            dtype,
            jnp.array(lin_tup, dtype=jnp.int32),
        )

    # ---------- Convenience ----------

    @property
    def nnz(self) -> int:
        """Number of stored (non-zero) entries."""
        return int(self.values.shape[0])

    def __matmul__(self, x: Array) -> Array:
        """Compute ``self @ x`` via :func:`sparse_matvec`."""
        return sparse_matvec(self, x)

    def __getitem__(self, ij: tuple[int, int]) -> Array:
        """Return the value at ``(i, j)``, or zero if the entry is not stored.

        Parameters
        ----------
        ij : (int, int)
            Row and column index.

        Returns
        -------
        Array
            Scalar value ``mat[i, j]``.

        """
        i, j = ij
        _, n_cols = self.shape
        target = i * n_cols + j
        idx = jnp.searchsorted(self.linear_indices, target)  # O(log nnz)
        found = (idx < self.nnz) & (self.linear_indices[jnp.minimum(idx, self.nnz - 1)] == target)
        result: Array = jax.lax.cond(
            found,
            lambda: self.values[idx],
            lambda: jnp.zeros((), dtype=self.dtype),
        )
        return result


# ---------- Conversion ----------


def to_sparse(dense: Array, *, tol: float = 0.0) -> SparseMatrix:
    """Convert a dense 2-D array to a :class:`SparseMatrix`.

    Parameters
    ----------
    dense : Array, shape (n, m)
        Dense matrix to convert.
    tol : float, default 0.0
        Entries with ``|v| <= tol`` are treated as zero and dropped.

    Returns
    -------
    SparseMatrix
        Sparse representation of ``dense``.

    """
    import numpy as np  # use numpy for the one-time index extraction

    dense_np = np.asarray(dense)
    mask = np.abs(dense_np) > tol
    row_idx, col_idx = np.nonzero(mask)
    vals = dense_np[row_idx, col_idx]
    dtype = jnp.result_type(dense)
    n_cols = dense_np.shape[1]
    linear_idx = row_idx * n_cols + col_idx  # already sorted (np.nonzero is row-major)
    return SparseMatrix(
        row_indices=jnp.array(row_idx, dtype=jnp.int32),
        col_indices=jnp.array(col_idx, dtype=jnp.int32),
        values=jnp.array(vals, dtype=dtype),
        shape=dense_np.shape,
        dtype=dtype,
        linear_indices=jnp.array(linear_idx, dtype=jnp.int32),
    )


# ---------- Sparse matrix-vector multiply ----------


def sparse_matvec(mat: SparseMatrix, x: Array) -> Array:
    """Sparse matrix–vector product  ``y = mat @ x``.

    Parameters
    ----------
    mat : SparseMatrix
        Sparse matrix with shape ``(n, m)``.
    x : Array, shape (m,)
        Dense input vector.

    Returns
    -------
    Array, shape (n,)
        Result ``y`` where ``y[i] = sum_j mat[i, j] * x[j]``.

    """
    n_rows, _ = mat.shape
    contributions = mat.values * x[mat.col_indices]  # gather
    return jax.ops.segment_sum(contributions, mat.row_indices, num_segments=n_rows)
