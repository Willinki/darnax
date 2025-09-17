"""Unit tests for LayerMap.

- Structure & invariants (sorted keys, diagonal policy)
- Read-only row/neighbor views
- PyTree behavior (flatten through modules, round-trip)
- Deterministic iteration order
- Gradient visibility to module params.

"""

from __future__ import annotations

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jax import tree_util as jtu

# Adjust these imports to match your package layout
from bionet.layer_maps.sparse import LayerMap
from bionet.modules.interfaces import AbstractModule

PRECISION = 1e-6
DIM_TREE = 3

logger = logging.getLogger(__name__)


# ---------------------------- Dummy module ----------------------------


def _is_param(x):
    # "parameters" = inexact (float/complex) JAX arrays
    return eqx.is_inexact_array(x)


def _sum_params(tree):
    # robust: only sum differentiable array leaves
    return sum(jnp.sum(a) for a in jtu.tree_leaves(tree) if _is_param(a))


def _tree_equalish(a, b, rtol=PRECISION, atol=PRECISION):
    # structural equality for non-arrays, allclose for arrays
    a_leaves, a_tdef = jtu.tree_flatten(a)
    b_leaves, b_tdef = jtu.tree_flatten(b)
    if a_tdef != b_tdef:
        return False
    for xa, xb in zip(a_leaves, b_leaves, strict=True):
        if eqx.is_array(xa) and eqx.is_array(xb):
            if not jnp.allclose(xa, xb, rtol=rtol, atol=atol):
                return False
        elif xa != xb:
            return False
    return True


class DummyModule(AbstractModule):
    """Minimal Equinox module used to test PyTree exposure via LayerMap."""

    w: Array
    j: int = eqx.field(static=True)

    def __init__(self, value: float):
        """Initialize a scalar parameter."""
        self.w = jnp.asarray(value, dtype=jnp.float32)
        self.j = 1

    @property
    def has_state(self) -> bool:
        """Pretend this module carries state; irrelevant for these tests."""
        return True

    def __call__(self, x: Array) -> Array:
        """Elementwise multiply by the scalar parameter."""
        return self.w * x

    def backward(self, x, y, y_hat):
        """Return dummy copy of the module (unused)."""
        return self


# ---------------------------- Fixtures/helpers ----------------------------


def _make_map_unsorted() -> dict[int, dict[int, DummyModule]]:
    """Unsorted rows/cols on purpose to test sorting behavior.

    Graph:
      row 1: cols {1, 0}
      row 0: cols {0}
    """
    return {
        1: {1: DummyModule(1.0), 0: DummyModule(0.5)},
        0: {0: DummyModule(2.0)},
    }


def _make_map_missing_diag() -> dict[int, dict[int, DummyModule]]:
    """Map where (1,1) is missing to test diagonal validation."""
    return {
        0: {0: DummyModule(1.0), 1: DummyModule(0.3)},
        1: {0: DummyModule(0.7)},  # (1,1) missing
    }


# ---------------------------- Core tests ----------------------------


def test_from_dict_sorts_rows_and_cols():
    """Rows/cols are sorted deterministically at construction."""
    lm = LayerMap.from_dict(_make_map_unsorted())
    assert lm.rows() == (0, 1)
    assert lm.cols_of(0) == (0,)
    assert lm.cols_of(1) == (0, 1)

    # Row-wise iteration is deterministic and read-only
    rows = list(lm.rows())
    assert rows == [0, 1]
    r_items = list(lm.row_items())
    assert [i for i, _ in r_items] == [0, 1]


def test_diagonal_validation_ok_and_error():
    """Diagonal enforcement raises if any (i,i) is missing."""
    # OK: diagonals present
    LayerMap.from_dict(_make_map_unsorted(), require_diagonal=True)

    # Error: missing a diagonal entry
    with pytest.raises(AttributeError):
        LayerMap.from_dict(_make_map_missing_diag(), require_diagonal=True)

    # Allow missing diagonal if disabled
    LayerMap.from_dict(_make_map_missing_diag(), require_diagonal=False)


def test_neighbors_and_row_getitem_are_read_only():
    """Neighbors/row views are read-only; single-edge access still works."""
    lm = LayerMap.from_dict(_make_map_unsorted())
    neigh = lm.neighbors(1)
    rowview = lm[1]

    with pytest.raises(TypeError):
        neigh[2] = DummyModule(3.0)  # MappingProxyType is read-only

    with pytest.raises(TypeError):
        rowview[2] = DummyModule(3.0)

    # Accessing individual module still works
    mod = lm[1, 0]
    assert isinstance(mod, DummyModule)
    assert jnp.asarray(mod.w).shape == ()


def test_contains_and_to_dict_copy():
    """Membership works and to_dict returns a deep copy (mutations don't affect LayerMap)."""
    lm = LayerMap.from_dict(_make_map_unsorted())
    assert (1, 0) in lm
    assert (0, 1) not in lm  # row 0 has only col 0

    d = lm.to_dict()
    # Mutating the copy must not affect the LayerMap
    d[1][2] = DummyModule(9.0)
    assert (1, 2) not in lm


def test_tree_flatten_unflatten_roundtrip():
    """Flatten/unflatten preserves structure and module leaves."""
    lm = LayerMap.from_dict(_make_map_unsorted())
    leaves, aux = lm.tree_flatten()
    lm2 = LayerMap.tree_unflatten(aux, leaves)

    assert lm.rows() == lm2.rows()
    for (i, j), mod in lm.edge_items():
        mod2 = lm2[i, j]
        assert eqx.tree_equal(mod, mod2)


def test_layermap_is_pytree_and_flattens_through_modules():
    """All module parameters appear among PyTree leaves (flatten-through semantics)."""
    lm = LayerMap.from_dict(_make_map_unsorted())

    # Default leaf policy: arrays are leaves, modules are not
    leaves = jtu.tree_leaves(lm)
    # Expect exactly three scalar parameters: (0,0), (1,0), (1,1)
    assert len(leaves) == DIM_TREE
    assert all(isinstance(x, jax.Array) for x in leaves)

    got = sorted([float(x) for x in leaves])
    assert got == sorted([2.0, 0.5, 1.0])


def test_partition_params_vs_static():
    """eqx.partition should classify module params as params (not static)."""
    lm = LayerMap.from_dict(_make_map_unsorted())
    params, static = eqx.partition(lm, eqx.is_inexact_array)
    p_leaves = jtu.tree_leaves(params)
    s_leaves = jtu.tree_leaves(static)
    assert len(p_leaves) == DIM_TREE
    assert all(isinstance(x, jax.Array) for x in p_leaves)
    assert all(not isinstance(x, jax.Array) for x in s_leaves)


def test_row_and_edge_items_are_deterministic():
    """Edge iteration is deterministic and row-major."""
    lm = LayerMap.from_dict(_make_map_unsorted())
    # Row-major edge order: (0,0), (1,0), (1,1)
    edges = list(lm.edge_items())
    assert [ij for ij, _ in edges] == [(0, 0), (1, 0), (1, 1)]


def test_getitem_errors_on_wrong_key_type():
    """Accessing with wrong key types raises helpful errors."""
    lm = LayerMap.from_dict(_make_map_unsorted())
    with pytest.raises(TypeError):
        _ = lm["not-an-int"]  # type: ignore[index]
    with pytest.raises(TypeError):
        _ = lm[(0,)]  # type: ignore[index]
    with pytest.raises(KeyError):
        _ = lm[3]  # missing row


# ---------------------------- Optional gradient sanity ----------------------------


def test_partition_combine_roundtrip():
    """eqx.partition + eqx.combine should reconstruct the original object."""
    lm = LayerMap.from_dict(_make_map_unsorted())

    dyn, sta = eqx.partition(lm, eqx.is_inexact_array)
    lm2 = eqx.combine(dyn, sta)

    assert _tree_equalish(lm, lm2)


def test_grads_flow_through_modules():
    """Grads reach module params inside LayerMap; only inexact arrays are differentiated."""
    lm = LayerMap.from_dict(_make_map_unsorted())

    def loss_fn(lm_):
        return _sum_params(lm_)

    val, grads = eqx.filter_value_and_grad(loss_fn)(lm)
    assert pytest.approx(float(val)) == (2.0 + 0.5 + 1.0)
    g_leaves = [g for g in jtu.tree_leaves(grads) if _is_param(g)]
    assert g_leaves, "no differentiable grads found"
    assert all(abs(float(g) - 1.0) < PRECISION for g in g_leaves)


def test_partition_places_only_params_in_dynamic_tree():
    """Dynamic tree should only expose inexact arrays as meaningful leaves."""
    lm = LayerMap.from_dict(_make_map_unsorted())
    dyn, sta = eqx.partition(lm, eqx.is_inexact_array)

    # dynamic leaves filtered by predicate must all satisfy the predicate
    dyn_param_leaves = [x for x in jtu.tree_leaves(dyn) if _is_param(x)]
    assert dyn_param_leaves, "no parameter leaves found in dynamic tree"
    assert all(_is_param(x) for x in dyn_param_leaves)

    # sanity: if we zero dynamic params and recombine, the param-sum goes to zero
    def zero_if_param(x):
        return jnp.zeros_like(x) if _is_param(x) else x

    dyn_zero = jtu.tree_map(zero_if_param, dyn)
    lm_zero = eqx.combine(dyn_zero, sta)

    assert float(_sum_params(lm_zero)) == 0.0


def test_filter_value_and_grad_matches_partition_predicate():
    """Gradients are taken w.r.t. exactly the same leaves we count in the loss."""
    lm = LayerMap.from_dict(_make_map_unsorted())

    def loss_fn(lm_):
        # build loss from the dynamic side explicitly to match the grad filter
        dyn, _ = eqx.partition(lm_, _is_param)
        return sum(jnp.sum(a) for a in jtu.tree_leaves(dyn) if _is_param(a))

    val, grads = eqx.filter_value_and_grad(loss_fn)(lm)

    # value matches direct param-sum
    assert pytest.approx(float(val)) == float(_sum_params(lm))

    # grads appear only at param positions (and are ones for a sum)
    g_param = [g for g in jtu.tree_leaves(grads) if _is_param(g)]
    g_nonparam = [g for g in jtu.tree_leaves(grads) if not _is_param(g)]
    assert g_param and all(abs(float(g) - 1.0) < PRECISION for g in g_param)
    # non-params either absent or trivially zero; ensure nothing sneaks in
    assert all((eqx.is_array(g) and g.size == 0) or not eqx.is_array(g) for g in g_nonparam)
