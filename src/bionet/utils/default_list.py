from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, MutableSequence
from typing import Any, Generic, TypeVar, overload

from jax.tree_util import register_pytree_node_class

T = TypeVar("T")


# Private unique marker for "logical default" slots.
class _DefaultSentinel:
    """Unique, unmaterialized placeholder for default-valued slots.

    Notes
    -----
    This sentinel marks positions that *logically* contain a default value but
    have not been materialized. It is replaced with an actual value only when
    read via ``__getitem__``/``to_list``. This keeps the underlying storage
    lightweight and preserves where defaults were implied.

    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<DEFAULT>"


_DEFAULT = _DefaultSentinel()


@register_pytree_node_class
class DefaultList(MutableSequence[T | None], Generic[T]):
    """Default-filling mutable list, registered as a JAX PyTree.

    Key behavior
    ------------
    - Assigning or inserting beyond the current length *fills gaps* with a
      default sentinel (materialized only when read).
    - Slicing returns another :class:`DefaultList` that preserves default slots.
    - Behaves like a normal ``MutableSequence`` for typing and basic list ops.
    - Participates in JAX PyTree flatten/unflatten.

    Parameters
    ----------
    initial : Iterable[T | None], optional
        Initial concrete values to store. Defaults are not inserted unless
        indices are explicitly extended by assignment/insert.
    default : T | None, optional
        Value returned when reading a default slot *if* ``default_factory`` is
        not provided. May be ``None``.
    default_factory : Callable[[], T | None], optional
        Zero-arg callable that produces the value for a default slot on read.
        Takes precedence over ``default``.
        **Note:** each read materializes a fresh value; defaults are not cached
        per-slot.

    Notes
    -----
    PyTree leaves are the underlying elements, including the sentinel; any JAX
    ``tree_map`` should account for non-numeric leaves if defaults are present.

    Public type
    -----------
    The public element type is ``T | None`` because materialized defaults may
    legitimately be ``None``.

    """

    # ---------- init ----------
    def __init__(
        self,
        initial: Iterable[T | None] | None = None,
        *,
        default: T | None = None,
        default_factory: Callable[[], T | None] | None = None,
    ):
        """Create a :class:`DefaultList`.

        Raises
        ------
        TypeError
            If ``default_factory`` is not callable.

        """
        if default_factory is not None and not callable(default_factory):
            raise TypeError("default_factory must be callable or None")
        self._data: list[Any] = list(initial) if initial is not None else []
        self._default: T | None = default
        self._default_factory: Callable[[], T | None] | None = default_factory

    # ---------- PyTree protocol ----------
    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Return children and aux data for JAX PyTree protocol.

        Returns
        -------
        tuple[tuple[Any, ...], tuple[Any, ...]]
            Children are the raw underlying items (sentinels preserved).
            Aux data contains ``(default, default_factory)``.

        """
        children = tuple(self._data)
        aux = (self._default, self._default_factory)
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[Any, ...],
        children: tuple[Any, ...],
    ) -> DefaultList[T]:
        """Rebuild from aux and children (JAX PyTree protocol)."""
        default, default_factory = aux
        out: DefaultList[T] = cls(default=default, default_factory=default_factory)
        out._data = list(children)
        return out

    # ---------- helpers ----------
    def _make_default(self) -> T | None:
        """Produce a default value for a single slot.

        Returns
        -------
        T | None
            Fresh value if ``default_factory`` is set; otherwise ``default``.

        Notes
        -----
        Each read creates a *new* object when ``default_factory`` is provided.

        """
        if self._default_factory is not None:
            return self._default_factory()
        return self._default  # may be None

    def _fill_to(self, n: int) -> None:
        """Ensure ``len(_data) >= n`` by appending default sentinels."""
        if n > len(self._data):
            self._data.extend([_DEFAULT] * (n - len(self._data)))

    def _norm_index_for_set(self, idx: int) -> int:
        """Normalize negative indices for item assignment.

        For non-negative indices, returns the index unchanged (allows extension).
        For negative indices, ensures the index is within current bounds.
        """
        if idx >= 0:
            return idx
        if -idx > len(self._data):
            raise IndexError("assignment index out of range")
        return len(self._data) + idx

    # ---------- abstract methods required by MutableSequence ----------
    def __len__(self) -> int:
        """Return number of stored slots (including default placeholders)."""
        return len(self._data)

    def __iter__(self) -> Iterator[T | None]:
        """Iterate over *materialized* values (defaults materialized on the fly)."""
        for raw in self._data:
            yield self._make_default() if raw is _DEFAULT else raw

    @overload
    def __getitem__(self, idx: int, /) -> T | None: ...
    @overload
    def __getitem__(self, idx: slice, /) -> MutableSequence[T | None]: ...

    def __getitem__(self, idx: int | slice, /) -> T | None | MutableSequence[T | None]:
        """Return a materialized value, or a sliced :class:`DefaultList`."""
        if isinstance(idx, slice):
            child: DefaultList[T] = DefaultList(
                default=self._default,
                default_factory=self._default_factory,
            )
            child._data = self._data[idx]  # preserve sentinels
            return child
        val = self._data[idx]  # supports negative indices
        return self._make_default() if val is _DEFAULT else val

    @overload
    def __setitem__(self, idx: int, value: T | None, /) -> None: ...
    @overload
    def __setitem__(self, idx: slice, value: Iterable[T | None], /) -> None: ...

    def __setitem__(
        self,
        idx: int | slice,
        value: T | None | Iterable[T | None],
        /,
    ) -> None:
        """Assign a value; extending past the end fills gaps with defaults.

        Slice assignment expects an iterable (like Python lists do).
        """
        if isinstance(idx, slice):
            values = list(value)  # type: ignore[arg-type]
            self._data[idx] = values
            return
        norm = self._norm_index_for_set(idx)
        self._fill_to(norm + 1)
        self._data[norm] = value

    @overload
    def __delitem__(self, idx: int, /) -> None: ...
    @overload
    def __delitem__(self, idx: slice, /) -> None: ...

    def __delitem__(self, idx: int | slice, /) -> None:
        """Delete an item or slice (mirrors Python list semantics)."""
        del self._data[idx]

    def insert(self, index: int, value: T | None) -> None:
        """Insert at ``index``; if beyond end, fill gap with defaults then append.

        Negative indices are clamped like Python's ``list.insert``.
        """
        if index < 0:
            index = max(0, len(self._data) + index)
        if index > len(self._data):
            self._fill_to(index)
            self._data.append(value)
        else:
            self._data.insert(index, value)

    # ---------- convenience methods ----------
    def append(self, value: T | None) -> None:
        """Append a value."""
        self._data.append(value)

    def extend(self, values: Iterable[T | None]) -> None:
        """Extend from an iterable."""
        self._data.extend(values)

    def to_list(self, *, filter_defaults: bool = False) -> list[T | None]:
        """Materialize to a plain Python list.

        Parameters
        ----------
        filter_defaults : bool, default False
            If True, drop default slots entirely. Otherwise materialize them.

        Returns
        -------
        list[T | None]
            A list containing either materialized defaults or only concrete values.

        """
        if filter_defaults:
            return [
                (v if v is not _DEFAULT else self._make_default())
                for v in self._data
                if v is not _DEFAULT
            ]
        return [(v if v is not _DEFAULT else self._make_default()) for v in self._data]

    def __repr__(self) -> str:
        """Debug representation showing materialized values."""
        shown = [(v if v is not _DEFAULT else self._make_default()) for v in self._data]
        return f"DefaultList(default={self._default}, data={shown})"
