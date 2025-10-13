# tests/test_layermap_apply_params_only.py
from __future__ import annotations

import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

# Adjust this import to your project layout:
from darnax.layer_maps.sparse import LayerMap
from darnax.utils.layermap_utils import layermap_apply  # noqa: F401

# Minimal alias so type hints remain readable in tests.
SCALE = 1.5


# -------------------------------
# Dummy modules to populate LayerMap
# -------------------------------
class DummySub(eqx.Module):
    """Simple implementation of a submodule."""

    proj: jax.Array  # array -> should be transformed
    scale: float  # static -> must remain unchanged
    tags: tuple[str, ...]  # static -> must remain unchanged


class DummyModule(eqx.Module):
    """Simple implementation of a module."""

    w: jax.Array  # array -> should be transformed
    b: jax.Array  # array -> should be transformed
    sub: DummySub  # nested module containing arrays + statics
    name: str  # static -> must remain unchanged
    meta: dict  # static (python object) -> must remain unchanged


# -------------------------------
# Fixtures
# -------------------------------
@pytest.fixture
def lmap_3x3() -> LayerMap:
    """3x3 LayerMap with unique values per (i,j) to make assertions crisp."""
    lmap = {}
    for i in range(3):
        row: dict[int, DummyModule] = {}
        for j in range(3):
            base = 10 * i + j
            mod = DummyModule(
                w=jnp.full((2, 2), base, dtype=jnp.float32),
                b=jnp.full((3,), 100 + base, dtype=jnp.float32),
                sub=DummySub(
                    proj=jnp.full((4,), 1000 + base, dtype=jnp.float32),
                    scale=SCALE,
                    tags=("keep", "me", "static"),
                ),
                name=f"mod-{i}-{j}",
                meta={"frozen": True, "idx": (i, j)},
            )
            row[j] = mod
        lmap[i] = row
    return LayerMap.from_dict(lmap)


@pytest.fixture
def lmap_3x3_snapshot(lmap_3x3: LayerMap) -> LayerMap:
    """Deep copy snapshot to verify no in-place mutations occur."""
    return copy.deepcopy(lmap_3x3)


# -------------------------------
# Helper checks
# -------------------------------
def _arrays_equal(a: jax.Array, b: jax.Array) -> bool:
    return bool(jnp.array_equal(a, b))


def _assert_module_arrays(mod: DummyModule, expected_w, expected_b, expected_proj):
    """Assert equality of arrays in modules."""
    assert _arrays_equal(mod.w, expected_w)
    assert _arrays_equal(mod.b, expected_b)
    assert _arrays_equal(mod.sub.proj, expected_proj)


def _assert_module_statics_unchanged(mod: DummyModule):
    """Assert statics in module is unchanges."""
    assert mod.sub.scale == SCALE
    assert mod.sub.tags == ("keep", "me", "static")
    assert isinstance(mod.name, str) and mod.name.startswith("mod-")
    assert mod.meta.get("frozen") is True
    assert isinstance(mod.meta.get("idx"), tuple)


# -------------------------------
# Tests
# -------------------------------
def test_diagonal_selection_arrays_only_transformed(lmap_3x3, lmap_3x3_snapshot):
    """Change diagonal and check its the only one affected."""
    # f: increment arrays by 1 (simple, JAX-friendly)
    f = lambda x: x + 1  # noqa: E731

    new_lmap = layermap_apply(f=f, select_idxs=lambda ij: ij[0] == ij[1], lmap=lmap_3x3)

    # Check: diagonal (i==j) changed by +1; off-diagonal untouched.
    for i in range(3):
        for j in range(3):
            old_mod = lmap_3x3_snapshot[i][j]
            new_mod = new_lmap[i][j]

            if i == j:
                _assert_module_arrays(
                    new_mod,
                    expected_w=old_mod.w + 1,
                    expected_b=old_mod.b + 1,
                    expected_proj=old_mod.sub.proj + 1,
                )
            else:
                _assert_module_arrays(
                    new_mod,
                    expected_w=old_mod.w,
                    expected_b=old_mod.b,
                    expected_proj=old_mod.sub.proj,
                )

            # Statics must always be untouched.
            _assert_module_statics_unchanged(new_mod)

            # Original must be unmodified (functional / no in-place changes).
            _assert_module_arrays(
                lmap_3x3[i][j],  # original structure
                expected_w=old_mod.w,
                expected_b=old_mod.b,
                expected_proj=old_mod.sub.proj,
            )
            _assert_module_statics_unchanged(lmap_3x3[i][j])


def test_upper_triangle_selection_including_nested_leaves(lmap_3x3, lmap_3x3_snapshot):
    """Apply to upper triangle and check nested modification."""
    f = lambda x: 2.0 * x  # noqa: E731

    new_lmap = layermap_apply(f=f, select_idxs=lambda ij: ij[0] < ij[1], lmap=lmap_3x3)

    for i in range(3):
        for j in range(3):
            old_mod = lmap_3x3_snapshot[i][j]
            new_mod = new_lmap[i][j]

            if i < j:
                _assert_module_arrays(
                    new_mod,
                    expected_w=2.0 * old_mod.w,
                    expected_b=2.0 * old_mod.b,
                    expected_proj=2.0 * old_mod.sub.proj,
                )
            else:
                _assert_module_arrays(
                    new_mod,
                    expected_w=old_mod.w,
                    expected_b=old_mod.b,
                    expected_proj=old_mod.sub.proj,
                )
            _assert_module_statics_unchanged(new_mod)


def test_select_none_noop(lmap_3x3, lmap_3x3_snapshot):
    """Check that nones are skipped."""
    f = lambda x: x - 123.0  # noqa: E731

    new_lmap = layermap_apply(f=f, select_idxs=lambda _: False, lmap=lmap_3x3)

    for i in range(3):
        for j in range(3):
            old_mod = lmap_3x3_snapshot[i][j]
            new_mod = new_lmap[i][j]

            _assert_module_arrays(
                new_mod,
                expected_w=old_mod.w,
                expected_b=old_mod.b,
                expected_proj=old_mod.sub.proj,
            )
            _assert_module_statics_unchanged(new_mod)


def test_select_all_every_array_transformed_statics_preserved(lmap_3x3, lmap_3x3_snapshot):
    """Check if statics is preserved."""
    f = lambda x: x.astype(jnp.float32) + jnp.ones_like(x, dtype=jnp.float32)  # noqa: E731

    new_lmap = layermap_apply(f=f, select_idxs=lambda _: True, lmap=lmap_3x3)

    for i in range(3):
        for j in range(3):
            old_mod = lmap_3x3_snapshot[i][j]
            new_mod = new_lmap[i][j]

            _assert_module_arrays(
                new_mod,
                expected_w=old_mod.w + 1.0,
                expected_b=old_mod.b + 1.0,
                expected_proj=old_mod.sub.proj + 1.0,
            )
            _assert_module_statics_unchanged(new_mod)


def test_dtype_and_shape_preserved_by_simple_transform(lmap_3x3):
    """Check dtypes and shapes are preserved by transforms."""
    # transform preserves dtype/shape (adds 0.0)
    f = lambda x: x + jnp.zeros_like(x)  # noqa: E731

    new_lmap = layermap_apply(f=f, select_idxs=lambda ij: ij[0] == ij[1], lmap=lmap_3x3)

    for i in range(3):
        for j in range(3):
            mod = new_lmap[i][j]
            # Shapes unchanged
            assert mod.w.shape == (2, 2)
            assert mod.b.shape == (3,)
            assert mod.sub.proj.shape == (4,)
            # Dtypes unchanged
            assert mod.w.dtype == jnp.float32
            assert mod.b.dtype == jnp.float32
            assert mod.sub.proj.dtype == jnp.float32


def test_function_is_jax_friendly(lmap_3x3, lmap_3x3_snapshot):
    """Check function can be jiited."""

    # Use a small nontrivial JAX-friendly function (affine + sin for variety).
    def f(x: jax.Array) -> jax.Array:
        return 0.5 * x + jnp.sin(x)

    # Not JITing the utility itself (it performs Python-structure work), but `f` is JAXable.
    new_lmap = layermap_apply(f=f, select_idxs=lambda ij: ij[0] >= ij[1], lmap=lmap_3x3)

    for i in range(3):
        for j in range(3):
            old_mod = lmap_3x3_snapshot[i][j]
            new_mod = new_lmap[i][j]
            if i >= j:
                _assert_module_arrays(
                    new_mod,
                    expected_w=f(old_mod.w),
                    expected_b=f(old_mod.b),
                    expected_proj=f(old_mod.sub.proj),
                )
            else:
                _assert_module_arrays(
                    new_mod,
                    expected_w=old_mod.w,
                    expected_b=old_mod.b,
                    expected_proj=old_mod.sub.proj,
                )
            _assert_module_statics_unchanged(new_mod)
