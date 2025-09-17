from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, overload

from jax.tree_util import register_pytree_node_class

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from bionet.modules.interfaces import AbstractModule

logger = logging.getLogger(__name__)


@register_pytree_node_class
@dataclass(frozen=True)
class LayerMap:
    """PyTree wrapper around a dict-of-dicts with static keys and *non-static* values.

    Design goals
    ------------
    - Maintain the clarity of a nested dict-of-dicts API.
    - Keep the *structure* (row/column keys and their order) static for JIT stability.
    - **Flatten through modules** so that arrays inside Equinox modules are visible to
      JAX/Optax (parameters get gradients and updates).
    - Prevent structural mutation after construction.

    Notes
    -----
    - Row keys are sorted once at initialization.
    - Column keys for each row are also sorted at initialization.
    - Keys are part of the treedef (static). Module *parameters* are leaves.
    - Public dict-like accessors return read-only mappings to avoid accidental
      structural mutation. Use `to_dict()` if you need a deep copy for external use.

    Public type
    -----------
    Values are typed as ``AbstractModule`` so that any subclass can be stored.

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
            Nested mapping from row -> (col -> module).
        require_diagonal : bool
            Enforce that (i, i) exists for any i present as either a row or a column.

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
        """Check for every row, also its diagonal value is present."""
        rows = set(data.keys())
        missing: list[int] = []
        for k in sorted(rows):
            if k not in data[k]:
                missing.append(k)
        if missing:
            raise AttributeError(f"Diagonal policy violated: missing (i, i) for {missing}")

    # ---------- PyTree protocol ----------

    def tree_flatten(self) -> tuple[tuple[AbstractModule, ...], tuple[Any, ...]]:
        """Deconstruct the tree. Does NOT flatten through modules."""
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
        """Reconstruct the tree based on aux and children provided by tree_flatten."""
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
        """Access row mapping or individual module.

        - `lm[i]` returns a read-only mapping of neighbors for row `i`.
        - `lm[i, j]` returns the module at edge `(i, j)`.
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
        """Return True if the module key[1] -> key[0] is present."""
        i, j = key
        return i in self._data and j in self._data[i]

    def rows(self) -> tuple[int, ...]:
        """All row indices in sorted order (static)."""
        return self._rows

    def cols_of(self, i: int) -> tuple[int, ...]:
        """All column indices of row `i` in sorted order (static for a given map)."""
        return tuple(self._data[i].keys())

    def neighbors(self, i: int) -> Mapping[int, AbstractModule]:
        """Read-only mapping of neighbors (col → module) for row `i`."""
        return MappingProxyType(self._data[i])

    def row_items(self) -> Iterable[tuple[int, Mapping[int, AbstractModule]]]:
        """Iterate `(row, neighbors)` with deterministic ordering and read-only views."""
        for i in self._rows:
            yield i, MappingProxyType(self._data[i])

    def edge_items(self) -> Iterable[tuple[tuple[int, int], AbstractModule]]:
        """Iterate over `((i, j), module)` in deterministic row-major order."""
        for i in self._rows:
            for j, v in self._data[i].items():
                yield (i, j), v

    def to_dict(self) -> dict[int, dict[int, AbstractModule]]:
        """Deep copy as a plain dict-of-dicts (mutable, not tied to this object)."""
        return {i: dict(self._data[i]) for i in self._rows}
