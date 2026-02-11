"""LayerMap: a static, PyTree-friendly adjacency of modules.

`LayerMap` wraps a nested dict-of-dicts that maps receiver rows ``i`` to
neighbors (columns) ``j`` → module, i.e. it represents edges ``(i, j)``.
The structure (row/column keys and their order) is **static** for JIT stability,
while the **values** (modules and their parameters) are dynamic PyTree leaves.

Design goals
------------
- Keep a clear nested dict API while making the structure part of the treedef.
- Flatten *through* Equinox/JAX modules so inner arrays are visible to JAX/Optax.
- Forbid structural mutation after construction (frozen dataclass, read-only views).

Conventions
-----------
- Rows and, within each row, columns are **sorted** once at construction.
- Keys are integers (layer indices). Edge ``(i, j)`` connects sender/neighbor
  ``j`` into receiver ``i`` (lower-triangular including the diagonal is common).
- The diagonal policy can be enforced: every row must have its ``(i, i)`` self-edge.

PyTree behavior
---------------
`tree_flatten` returns all modules in deterministic row-major order as *children*,
plus static aux data describing the key layout. JAX/Equinox then flattens
module parameters further, so optimizers and transforms "see" the arrays inside.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, overload

from jax.tree_util import register_pytree_node_class

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from darnax.modules.interfaces import AbstractModule

logger = logging.getLogger(__name__)


@register_pytree_node_class
@dataclass(frozen=True)
class LayerMap:
    """PyTree wrapper around a dict-of-dicts with static keys and *non-static* values.

    Parameters
    ----------
    _data : dict[int, dict[int, AbstractModule]]
        Internal mapping from row ``i`` → (col ``j`` → module). Keys are sorted.
    _rows : tuple[int, ...]
        All row keys in sorted order (becomes part of the treedef).
    _ndim : int, default 2
        Tuple key arity (``(i, j)``). Exposed mainly for reconstruction.

    Notes
    -----
    - The dataclass is **frozen** to prevent structural mutation after creation.
      Use :meth:`to_dict` to obtain a mutable deep copy if you truly need one.
    - Read-only accessors (e.g., :meth:`neighbors`) return ``MappingProxyType``.
    - The keys (rows/cols) are included in the PyTree aux data, so the layout
      is static under JIT; values (modules) are the dynamic leaves.

    Public type
    -----------
    Values are typed as :class:`~darnax.modules.interfaces.AbstractModule` so any
    concrete layer/adapter subtype can be stored.

    """

    _data: dict[int, dict[int, AbstractModule]]
    _rows: tuple[int, ...]
    _ndim: int = 2

    # ---------- Constructors ----------

    @staticmethod
    def from_dict(
        mapping: Mapping[int, Mapping[int, AbstractModule]],
        *,
        require_diagonal: bool = True,
    ) -> LayerMap:
        """Construct a LayerMap from a nested mapping.

        Parameters
        ----------
        mapping : Mapping[int, Mapping[int, AbstractModule]]
            Nested mapping from row ``i`` → (col ``j`` → module).
        require_diagonal : bool, default True
            If ``True``, enforce that each present row ``i`` has an explicit
            self-edge ``(i, i)``.

        Returns
        -------
        LayerMap
            A new instance with rows and per-row columns sorted deterministically.

        Raises
        ------
        AttributeError
            If ``require_diagonal=True`` and some ``(i, i)`` is missing.

        """
        rows: tuple[int, ...] = tuple(sorted(mapping.keys()))
        data: dict[int, dict[int, AbstractModule]] = {}
        for i in rows:
            # Ensure deterministic column order per row.
            cols_sorted: dict[int, AbstractModule] = dict(sorted(mapping[i].items()))
            data[i] = cols_sorted
        if require_diagonal:
            LayerMap._validate_diagonal(data)
        return LayerMap(data, rows)

    @staticmethod
    def _validate_diagonal(data: Mapping[int, Mapping[int, AbstractModule]]) -> None:
        """Validate that every row has its diagonal entry.

        Parameters
        ----------
        data : Mapping[int, Mapping[int, AbstractModule]]
            Row → (col → module) mapping.

        Raises
        ------
        AttributeError
            If any row ``i`` lacks the diagonal key ``i``.

        """
        rows = set(data.keys())
        missing: list[int] = []
        for k in sorted(rows):
            if k not in data[k]:
                missing.append(k)
        if missing:
            raise AttributeError(f"Diagonal policy violated: missing (i, i) for {missing}")

    # ---------- PyTree protocol ----------

    def tree_flatten(self) -> tuple[tuple[AbstractModule, ...], tuple[Any, ...]]:
        """Deconstruct into children and aux (PyTree protocol).

        Returns
        -------
        children : tuple[AbstractModule, ...]
            Modules in deterministic row-major order; JAX/Equinox will flatten
            their parameter fields further.
        aux : tuple[Any, ...]
            Static metadata: ``(rows, cols_per_row, ndim)`` to reconstruct the treedef.

        Notes
        -----
        We intentionally **do not** include keys as children; keys are part of
        the static aux data so JIT sees a stable structure even when values change.

        """
        rows = self._rows
        cols_per_row = tuple(tuple(self._data[i].keys()) for i in rows)
        children: list[AbstractModule] = []
        for i, cols in zip(rows, cols_per_row, strict=True):
            for j in cols:
                children.append(self._data[i][j])
        aux = (rows, cols_per_row, self._ndim)
        return tuple(children), aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[Any, ...],
        children: Iterable[AbstractModule],
    ) -> LayerMap:
        """Reconstruct from aux and children (PyTree protocol).

        Parameters
        ----------
        aux : tuple[Any, ...]
            The static metadata returned by :meth:`tree_flatten`:
            ``(rows, cols_per_row, ndim)``.
        children : Iterable[AbstractModule]
            Modules in the exact row-major order produced by :meth:`tree_flatten`.

        Returns
        -------
        LayerMap
            A new instance with the same static key layout and provided values.

        """
        rows, cols_per_row, ndim = aux
        it = iter(children)
        data: dict[int, dict[int, AbstractModule]] = {}
        for i, cols in zip(rows, cols_per_row, strict=True):
            row_dict: dict[int, AbstractModule] = {}
            for j in cols:
                row_dict[j] = next(it)
            data[i] = row_dict
        return cls(data, rows, ndim)

    # ---------- Dict-like API (read-only for structure) ----------

    @overload
    def __getitem__(self, i: int) -> Mapping[int, AbstractModule]: ...

    @overload
    def __getitem__(self, ij: tuple[int, int]) -> AbstractModule: ...

    def __getitem__(
        self, key: int | tuple[int, int]
    ) -> AbstractModule | Mapping[int, AbstractModule]:
        """Access a row mapping or a single edge.

        Parameters
        ----------
        key : int or (int, int)
            - ``lm[i]`` returns a read-only mapping of neighbors for row ``i``.
            - ``lm[i, j]`` returns the module at edge ``(i, j)``.

        Returns
        -------
        Mapping[int, AbstractModule] or AbstractModule
            A read-only row mapping, or the concrete module at the given edge.

        Raises
        ------
        TypeError
            If the key is neither ``int`` nor ``(int, int)`` with the expected arity.

        """
        if isinstance(key, int):
            return MappingProxyType(self._data[key])
        if (
            isinstance(key, tuple)
            and len(key) == self._ndim
            and isinstance(key[0], int)
            and isinstance(key[1], int)
        ):
            i, j = key
            return self._data[i][j]
        raise TypeError("Key must be int (row) or tuple[int, int] (edge)")

    def __contains__(self, key: tuple[int, int]) -> bool:  # edge membership
        """Return ``True`` if an edge exists.

        Parameters
        ----------
        key : (int, int)
            The edge ``(i, j)`` to test.

        Returns
        -------
        bool
            ``True`` if ``(i, j)`` is present, ``False`` otherwise.

        """
        i, j = key
        return i in self._data and j in self._data[i]

    def rows(self) -> tuple[int, ...]:
        """All row indices in sorted order (static)."""
        return self._rows

    def cols_of(self, i: int) -> tuple[int, ...]:
        """All column indices of row `i` in sorted order (static for a given map)."""
        return tuple(self._data[i].keys())

    def neighbors(self, i: int) -> Mapping[int, AbstractModule]:
        """Read-only mapping of neighbors (``col → module``) for row ``i``."""
        return MappingProxyType(self._data[i])

    def row_items(
        self,
        skip_last: bool = False,
        subset: Literal["backward", "forward", "all", "inference"] = "all",
    ) -> Iterable[tuple[int, Mapping[int, AbstractModule]]]:
        """Iterate over rows with deterministic ordering and read-only views.

        Parameters
        ----------
        skip_last : bool, default False
            If ``True``, omit the last receiver row (useful when the output row
            is sink-only).
        subset : ["backward", "forward", "all"], default "all
            If ``forward``, keep only edges ``(i, j)`` with ``j <= i`` (i.e.,
            lower-triangular including the diagonal), which is a common
            “feed-forward” scheduling constraint.
            If ``backward``, keep only edges ``(i, j)`` with ``j >= i`` (i.e.,
            upper-triangular including the diagonal).
            If ``inference``, keep only edges ``(i, j)`` with ``j != last`` (i.e.,
            everything except last column).


        Yields
        ------
        (row, neighbors) : tuple[int, Mapping[int, AbstractModule]]
            The row index and a read-only mapping of its neighbors.

        """
        row_keys = list(self._rows)
        output_idx: int = row_keys[-1]
        if skip_last:
            row_keys = row_keys[:-1]
        for r_idx in row_keys:
            data = self._data[r_idx]
            if subset == "forward":
                data = {k: v for k, v in data.items() if k <= r_idx}
            elif subset == "backward":
                data = {k: v for k, v in data.items() if k >= r_idx}
            elif subset == "inference":
                data = {k: v for k, v in data.items() if k != output_idx}
            else:
                raise AttributeError
            yield r_idx, MappingProxyType(data)

    def edge_items(self) -> Iterable[tuple[tuple[int, int], AbstractModule]]:
        """Iterate over edges in deterministic row-major order.

        Yields
        ------
        ((i, j), module) : tuple[tuple[int, int], AbstractModule]
            Edge key and its module.

        """
        for i in self._rows:
            for j, v in self._data[i].items():
                yield (i, j), v

    def to_dict(self) -> dict[int, dict[int, AbstractModule]]:
        """Deep copy as a plain, mutable dict-of-dicts.

        Returns
        -------
        dict[int, dict[int, AbstractModule]]
            A new mapping with the same keys and module values. Mutations on
            this result do **not** affect the original ``LayerMap``.

        """
        return {i: dict(self._data[i]) for i in self._rows}
