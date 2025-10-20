"""Default-filling list registered as a JAX PyTree.

``DefaultList`` behaves like a Python ``list`` but supports *logical defaults*:
gaps created by assigning/inserting past the current end are represented by a
sentinel and **materialized only on read**. This keeps storage compact while
preserving where defaults were implied.

Features
--------
- Works like ``collections.abc.MutableSequence`` (indexing, slicing, insert, del).
- Extending past the end fills with a sentinel; the actual default value is
  produced only when accessed (``__getitem__``/``__iter__``/``to_list``).
- Slicing returns another ``DefaultList`` that **preserves** sentinel slots.
- Registered as a JAX **PyTree** (``tree_flatten``/``tree_unflatten``).

Notes
-----
- If ``default_factory`` is provided, **every read** of a default slot creates
  a fresh value (no per-slot caching).
- Leaves in the PyTree include the sentinel; JAX transforms that map over
  leaves should tolerate non-numeric entries when defaults are present.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, MutableSequence
from typing import Any, Generic, TypeVar, overload

from jax.tree_util import register_pytree_node_class

T = TypeVar("T")


# Private unique marker for "logical default" slots.
class _DefaultSentinel:
    """Unique, unmaterialized placeholder for default-valued slots.

    This sentinel marks positions that *logically* contain a default value but
    have not been materialized. It is replaced with an actual value only when
    read via ``__getitem__``, ``__iter__``, or :meth:`DefaultList.to_list`.

    Notes
    -----
    The sentinel itself may appear as a **leaf** in a PyTree during flattening.

    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<DEFAULT>"


_DEFAULT = _DefaultSentinel()


@register_pytree_node_class
class DefaultList(MutableSequence[T | None], Generic[T]):
    """Default-filling mutable list, registered as a JAX PyTree.

    Assigning or inserting beyond the current length *fills gaps* with a
    sentinel; the concrete default value is produced only when read. Slicing
    returns another :class:`DefaultList` that preserves sentinel slots.

    Parameters
    ----------
    initial : Iterable[T | None], optional
        Concrete values to seed the list. Defaults are not inserted unless
        indices are explicitly extended by assignment/insert.
    default : T | None, optional
        Value returned for a default slot *if* ``default_factory`` is not set.
    default_factory : Callable[[], T | None], optional
        Zero-arg callable producing the value for a default slot on read.
        Takes precedence over ``default``. Each read yields a **fresh** value.

    Notes
    -----
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
        """Return children and aux data for the JAX PyTree protocol.

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
        """Rebuild from aux and children (JAX PyTree protocol).

        Parameters
        ----------
        aux : tuple[Any, ...]
            The auxiliary data returned by :meth:`tree_flatten`.
        children : tuple[Any, ...]
            The raw items (may include sentinels).

        Returns
        -------
        DefaultList[T]
            A reconstructed list with identical contents and defaults.

        """
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
        A new object is returned on each call when ``default_factory`` is used.

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

        Raises
        ------
        IndexError
            If a negative index refers before the start of the list.

        """
        if idx >= 0:
            return idx
        if -idx > len(self._data):
            raise IndexError("assignment index out of range")
        return len(self._data) + idx

    # ---------- abstract methods required by MutableSequence ----------
    def __len__(self) -> int:
        """Return the number of stored slots (including default placeholders)."""
        return len(self._data)

    def __iter__(self) -> Iterator[T | None]:
        """Iterate over **materialized** values.

        Yields
        ------
        T | None
            Concrete values where sentinel positions are replaced by defaults.

        Notes
        -----
        When a ``default_factory`` is set, repeated iterations can produce
        distinct objects for the same default slot (no caching).

        """
        for raw in self._data:
            yield self._make_default() if raw is _DEFAULT else raw

    @overload
    def __getitem__(self, idx: int, /) -> T | None: ...

    @overload
    def __getitem__(self, idx: slice, /) -> MutableSequence[T | None]: ...

    def __getitem__(self, idx: int | slice, /) -> T | None | MutableSequence[T | None]:
        """Return a materialized value or a sliced :class:`DefaultList`.

        Parameters
        ----------
        idx : int or slice
            Index or slice.

        Returns
        -------
        T | None or MutableSequence[T | None]
            Single value (defaults materialized) or a sliced ``DefaultList``
            that **preserves** sentinel positions.

        """
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

        Parameters
        ----------
        idx : int or slice
            Index or slice to assign into.
        value : T | None or Iterable[T | None]
            Value(s) to assign. Slice assignment follows Python list semantics.

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

        Parameters
        ----------
        index : int
            Insertion position. Negative indices are clamped like
            ``list.insert``.
        value : T | None
            Value to insert.

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
            If ``True``, drop default slots entirely (positions are lost).
            Otherwise, materialize defaults into concrete values.

        Returns
        -------
        list[T | None]
            Either a list containing concrete values *and* materialized defaults,
            or (if ``filter_defaults=True``) only the explicitly set values.

        Notes
        -----
        With ``default_factory``, materialized defaults are freshly created and
        are not cached per-slot.

        """
        if filter_defaults:
            return [
                (v if v is not _DEFAULT else self._make_default())
                for v in self._data
                if v is not _DEFAULT
            ]
        return [(v if v is not _DEFAULT else self._make_default()) for v in self._data]

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        """Debug representation with materialized values."""
        shown = [(v if v is not _DEFAULT else self._make_default()) for v in self._data]
        return f"DefaultList(default={self._default}, data={shown})"
