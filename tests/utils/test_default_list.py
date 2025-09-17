"""Tests for DefaultList (default-filling mutable list registered as a PyTree)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bionet.utils.default_list import DefaultList


@pytest.mark.parametrize(
    "make_list,expected_default",
    [
        (lambda: DefaultList(default=0), 0),
        (lambda: DefaultList(default=None), None),
        (lambda: DefaultList(default_factory=lambda: 0), 0),
    ],
)
def test_set_beyond_end_fills_with_defaults(make_list, expected_default):
    """Assign past the end fills defaults in the gap; reading materializes them."""
    dl = make_list()
    dl[2] = 5  # fills [<DEFAULT>, <DEFAULT>, 5]
    n1, n2 = 3, 5
    assert len(dl) == n1
    assert dl[0] == expected_default
    assert dl[1] == expected_default
    assert dl[2] == n2
    # Iteration materializes defaults too
    assert list(dl) == [expected_default, expected_default, 5]


def test_default_factory_returns_fresh_objects_each_read():
    """Each read of a default slot uses a *fresh* object from the factory."""
    dl = DefaultList(default_factory=list)
    dl[2] = "X"
    a = dl[0]
    b = dl[1]
    assert a == [] and b == []
    assert a is not b  # fresh objects, not shared


def test_append_and_extend_and_len():
    """Basic list-like ops work and count default placeholders too."""
    dl = DefaultList(default=0)
    n = 3
    dl.append(1)
    dl.extend([2, 3])
    assert len(dl) == n
    assert dl.to_list() == [1, 2, 3]


def test_slice_get_returns_defaultlist_and_preserves_defaults():
    """Slicing returns DefaultList with sentinels preserved (defaults still implicit)."""
    dl = DefaultList(default=0, initial=[10])
    dl[3] = 4  # -> [10, <D>, <D>, 4]
    sl = dl[1:4]  # type: ignore[index]
    assert isinstance(sl, DefaultList)
    # Reading from the slice materializes defaults with the same default spec
    assert sl.to_list() == [0, 0, 4]


def test_slice_assign_replaces_underlying_region():
    """Slice assignment behaves like list; requires an iterable."""
    dl = DefaultList(initial=[1, 2, 3], default=0)
    dl[1:3] = [9, 8]
    assert dl.to_list() == [1, 9, 8]
    with pytest.raises(TypeError):
        dl[0:1] = 42  # type: ignore[assignment]


def test_insert_beyond_end_fills_then_appends():
    """Insert past the end fills the gap with defaults then appends the value."""
    n1, n2 = 5, 7
    dl = DefaultList(default=0)
    dl.insert(n1, n2)
    assert len(dl) == n1 + 1
    assert dl.to_list() == [0, 0, 0, 0, 0, 7]


def test_negative_indexing_and_assignment_bounds():
    """Negative indices behave like list; out-of-range negative set raises."""
    dl = DefaultList(initial=[10, 20, 30], default=0)
    dl[-1] = 99
    assert dl.to_list() == [10, 20, 99]
    with pytest.raises(IndexError):
        dl[-4] = 1  # out of range for current length 3


def test_delete_int_and_slice():
    """Deleting elements works with ints and slices like Python lists."""
    dl = DefaultList(initial=[1, 2, 3, 4], default=0)
    del dl[1]
    assert dl.to_list() == [1, 3, 4]
    del dl[1:3]
    assert dl.to_list() == [1]


def test_to_list_filter_defaults():
    """Materialization vs filtering of default slots."""
    dl = DefaultList(default=0)
    dl[3] = 4  # [0,0,0,4]
    assert dl.to_list(filter_defaults=False) == [0, 0, 0, 4]
    assert dl.to_list(filter_defaults=True) == [4]


def test_repr_shows_materialized_values():
    """__repr__ includes default value and materialized data."""
    dl = DefaultList(default=0)
    dl[2] = 5
    s = repr(dl)
    assert "DefaultList" in s and "default=0" in s and "data=[0, 0, 5]" in s


def test_pytree_roundtrip_flatten_unflatten():
    """Flatten/unflatten via JAX PyTree protocol preserves structure and defaults."""
    dl = DefaultList(default=0)
    dl[3] = 4
    children, aux = dl.tree_flatten()
    rebuilt = DefaultList.tree_unflatten(aux, children)
    # Reading both should materialize the same data
    assert rebuilt.to_list() == dl.to_list()


def test_pytree_tree_map_identity_and_numeric_map():
    """tree_map keeps the structure and can transform numeric leaves."""
    dl = DefaultList(default=0, initial=[1])
    dl[2] = 3  # -> [1, <D>, 3]
    # Identity map (safe even with sentinels)
    dl_id = jax.tree_util.tree_map(lambda x: x, dl)
    assert isinstance(dl_id, DefaultList)
    assert dl_id.to_list() == dl.to_list()

    # Numeric-only map: add 1 to numbers; leave others (e.g., sentinel) untouched.
    def add_one_numbers(x):
        return x + 1 if isinstance(x, int | float | jnp.ndarray) else x

    dl_inc = jax.tree_util.tree_map(add_one_numbers, dl)
    # Defaults are sentinels, so only concrete numbers were incremented
    assert dl_inc.to_list() == [2, 0, 4]  # default still 0 upon read
