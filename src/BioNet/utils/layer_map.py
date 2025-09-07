from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Tuple, overload
from jax.tree_util import register_pytree_node_class

from src.utils.default_list import DefaultList
from src.modules.interfaces import AbstractModule


@register_pytree_node_class
class LayerMap:
    """
    Wrapper around a list of DefaultList[AbstractModule].

    - Default value is None.
    - Elements must be AbstractModule (or subclasses) when setting.
    - Supports lm[i] -> DefaultList[AbstractModule]
               lm[i, j] -> AbstractModule | None  (materialized default)
    - Registered as a JAX PyTree (classic tree_flatten/unflatten).
    """

    __slots__ = ("data",)

    def __init__(self, layers: int):
        if not isinstance(layers, int) or layers <= 0:
            raise ValueError(f"`layers` must be a positive int; got {layers}")
        # Each row defaults to None; elements will be AbstractModule instances.
        self.data: List[DefaultList[AbstractModule]] = [
            DefaultList[AbstractModule](default=None) for _ in range(layers)
        ]

    # ---------- PyTree protocol ----------
    def tree_flatten(self):
        # Each row is a child pytree (DefaultList is registered already).
        children = tuple(self.data)
        aux = None  # no static aux needed
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        # Bypass __init__ to avoid re-alloc; restore directly.
        obj = cls.__new__(cls)  # type: ignore
        obj.data = list(children)
        return obj

    # ---------- list-like convenience ----------
    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[DefaultList[AbstractModule]]:
        return iter(self.data)

    @overload
    def __getitem__(self, idx: int) -> DefaultList[AbstractModule]: ...
    @overload
    def __getitem__(self, idx: Tuple[int, int]) -> Optional[AbstractModule]: ...

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            layer, pos = idx
            return self.data[layer][pos]  # DefaultList materializes None for gaps
        return self.data[idx]

    @overload
    def __setitem__(self, idx: int, value: DefaultList[AbstractModule]) -> None: ...
    @overload
    def __setitem__(self, idx: Tuple[int, int], value: AbstractModule) -> None: ...

    def __setitem__(self, idx, value) -> None:
        if isinstance(idx, tuple):
            layer, pos = idx
            if not isinstance(value, AbstractModule):
                raise TypeError(
                    f"Expected AbstractModule at [{layer},{pos}], got {type(value).__name__}"
                )
            self.data[layer][pos] = value
        else:
            if not isinstance(value, DefaultList):
                raise TypeError(
                    f"Row must be a DefaultList[AbstractModule]; got {type(value).__name__}"
                )
            # light runtime check on contents (best-effort; not deep)
            for v in value.to_list(filter_defaults=True):
                if not isinstance(v, AbstractModule):
                    raise TypeError(
                        "All non-default row elements must be AbstractModule instances"
                    )
            self.data[idx] = value

    # Row helpers
    def append_row(self) -> None:
        self.data.append(DefaultList[AbstractModule](default=None))

    def insert_row(self, index: int) -> None:
        if index < 0:
            index = max(0, len(self.data) + index)
        if index > len(self.data):
            # Fill with empty rows up to index
            self.data.extend(
                DefaultList[AbstractModule](default=None)
                for _ in range(index - len(self.data))
            )
        self.data.insert(index, DefaultList[AbstractModule](default=None))

    # Matrix materialization
    def to_matrix(
        self, *, filter_defaults: bool = False
    ) -> List[List[Optional[AbstractModule]]]:
        """
        Returns a nested Python list. If filter_defaults=True, default slots are dropped
        from each row; otherwise defaults are materialized as None.
        """
        return [row.to_list(filter_defaults=filter_defaults) for row in self.data]

    def __repr__(self) -> str:
        rows = ", ".join(repr(row) for row in self.data)
        return f"LayerMap([{rows}])"
