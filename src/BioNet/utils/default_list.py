from __future__ import annotations

from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    overload,
)
from collections.abc import MutableSequence
from jax.tree_util import register_pytree_node_class

T = TypeVar("T")


# Private unique marker for "logical default" slots.
class _DefaultSentinel:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<DEFAULT>"


_DEFAULT = _DefaultSentinel()


@register_pytree_node_class
class DefaultList(MutableSequence[T], Generic[T]):
    """
    Default-filling mutable list, registered as a JAX PyTree.

    - Assigning or inserting beyond the current length fills missing slots with a default.
    - Defaults are tracked via a sentinel; actual defaults are materialized only on read.
    - Slicing returns another DefaultList (preserving default slots).
    - Subclasses MutableSequence so type checkers treat it like a normal list.

    Parameters
    ----------
    initial : iterable of T (optional)
    default : T used when materializing default slots (optional; can be None)
    default_factory : callable producing T; takes precedence over `default`
    """

    # ---------- init ----------
    def __init__(
        self,
        initial: Optional[Iterable[T]] = None,
        *,
        default: Optional[T] = None,
        default_factory: Optional[Callable[[], T]] = None,
    ):
        if default_factory is not None and not callable(default_factory):
            raise TypeError("default_factory must be callable or None")
        self._data: List[Any] = list(initial) if initial is not None else []
        self._default: Optional[T] = default
        self._default_factory: Optional[Callable[[], T]] = default_factory

    # ---------- PyTree protocol ----------
    def tree_flatten(self):
        children = tuple(self._data)  # leaves (sentinels remain as-is)
        aux = (self._default, self._default_factory)  # static aux data
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        default, default_factory = aux
        out = cls(default=default, default_factory=default_factory)
        out._data = list(children)
        return out

    # ---------- helpers ----------
    def _make_default(self) -> T:
        if self._default_factory is not None:
            return self._default_factory()
        return self._default  # may be None

    def _fill_to(self, n: int) -> None:
        """Ensure len(_data) >= n by appending sentinel slots."""
        if n > len(self._data):
            self._data.extend([_DEFAULT] * (n - len(self._data)))

    def _norm_index_for_set(self, idx: int) -> int:
        """Normalize negative indices like list, but allow extending on positive idx."""
        if idx >= 0:
            return idx
        # negative index must be within current bounds
        if -idx > len(self._data):
            raise IndexError("assignment index out of range")
        return len(self._data) + idx

    # ---------- abstract methods required by MutableSequence ----------
    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self._data)):
            yield self[i]

    @overload
    def __getitem__(self, idx: int) -> T: ...
    @overload
    def __getitem__(self, idx: slice) -> "DefaultList[T]": ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            child = DefaultList[T](
                default=self._default, default_factory=self._default_factory
            )
            child._data = self._data[idx]  # preserve sentinels
            return child
        val = self._data[idx]  # supports negative indexing per list semantics
        return self._make_default() if val is _DEFAULT else val

    def __setitem__(self, idx, value) -> None:
        if isinstance(idx, slice):
            # Slice assign: ensure value is iterable and assign directly to underlying list.
            values = list(value)  # will raise TypeError if not iterable
            self._data[idx] = values
            return
        idx = self._norm_index_for_set(idx)
        self._fill_to(idx + 1)
        self._data[idx] = value

    def __delitem__(self, idx) -> None:
        del self._data[idx]  # supports int or slice deletions

    def insert(self, index: int, value: T) -> None:
        # Insert beyond end -> fill gap, then append (Python list would clamp to end; we fill)
        if index < 0:
            # list.insert handles negative indexes by clamping; emulate that
            index = max(0, len(self._data) + index)
        if index > len(self._data):
            self._fill_to(index)
            self._data.append(value)
        else:
            self._data.insert(index, value)

    # ---------- convenience methods ----------
    def append(self, value: T) -> None:
        self._data.append(value)

    def extend(self, values: Iterable[T]) -> None:
        self._data.extend(values)

    def to_list(self, *, filter_defaults: bool = False) -> List[T]:
        """Materialize to a Python list. If filter_defaults=True, drop default slots."""
        if filter_defaults:
            return [
                (v if v is not _DEFAULT else self._make_default())
                for v in self._data
                if v is not _DEFAULT
            ]
        else:
            return [
                (v if v is not _DEFAULT else self._make_default()) for v in self._data
            ]

    def __repr__(self) -> str:
        shown = [(v if v is not _DEFAULT else self._make_default()) for v in self._data]
        return f"DefaultList(default={self._default}, data={shown})"
