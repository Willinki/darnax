# ruff: noqa: D100,D101,D102,D103
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from bionet.layer_maps.sparse import LayerMap
from bionet.modules.debug import DebugAdapter, DebugLayer
from bionet.orchestrators.sequential import SequentialOrchestrator
from bionet.states.sequential import SequentialState

if TYPE_CHECKING:
    from collections.abc import Iterable

KeyArray = jax.Array


# --------------------------------- Helpers ----------------------------------


def _assert_same_treedef(a: Any, b: Any) -> None:
    leaves_a, td_a = jax.tree.flatten(a)
    leaves_b, td_b = jax.tree.flatten(b)
    assert td_a == td_b
    assert len(leaves_a) == len(leaves_b)
    for la, lb in zip(leaves_a, leaves_b, strict=True):
        assert isinstance(la, jnp.ndarray) and isinstance(lb, jnp.ndarray)
        assert la.shape == lb.shape and la.dtype == lb.dtype


def _assert_state_equal(s1: SequentialState, s2: SequentialState) -> None:
    # Compare elementwise using indexing; assumes __len__/__getitem__ exist.
    assert len(s1) == len(s2)
    for i in range(len(s1)):
        assert jnp.allclose(s1[i], s2[i])


def _available_steps(orch: SequentialOrchestrator) -> Iterable[str]:
    # Prefer canonical names; include aliases if present.
    for name in ("step", "step_inference", "step_forward"):
        if hasattr(orch, name):
            yield name


# --------------------------------- Fixtures ---------------------------------


@pytest.fixture
def key() -> KeyArray:
    return jax.random.PRNGKey(0)


@pytest.fixture
def tiny_graph(key: KeyArray) -> tuple[SequentialOrchestrator, SequentialState, KeyArray]:
    """2x2 graph with both directions populated.

    lmap:
      [0,0] diag, [0,1] (1->0)
      [1,0] (0->1), [1,1] diag
    """
    diag0 = DebugLayer()
    diag1 = DebugLayer()
    diag2 = DebugLayer()
    diag3 = DebugLayer()
    e01 = DebugAdapter()
    e02 = DebugAdapter()
    e10 = DebugAdapter()
    e12 = DebugAdapter()
    e20 = DebugAdapter()
    e21 = DebugAdapter()
    e12 = DebugAdapter()
    e23 = DebugAdapter()
    e32 = DebugAdapter()
    e31 = DebugAdapter()
    e30 = DebugAdapter()

    lmap_dict = {
        0: {0: diag0, 1: e01, 2: e02},
        1: {0: e10, 1: diag1, 2: e12},
        2: {0: e20, 1: e21, 2: diag2, 3: e23},
        3: {0: e30, 1: e31, 2: e32, 3: diag3},
    }
    lmap = LayerMap.from_dict(lmap_dict, require_diagonal=True)
    state = SequentialState(sizes=(3, 3, 3, 3))  # batch/shape handled by your implementation
    orch = SequentialOrchestrator(layers=lmap)
    return orch, state, key


# ---------------------------------- Tests -----------------------------------


def test_pytree_flatten_roundtrip(
    tiny_graph: tuple[SequentialOrchestrator, SequentialState, KeyArray],
) -> None:
    orch, _, _ = tiny_graph
    leaves, treedef = jax.tree.flatten(orch)
    rebuilt = jax.tree.unflatten(treedef, leaves)
    # Same structure + leaf shapes/dtypes
    _assert_same_treedef(orch, rebuilt)


def test_eqx_partition_roundtrip_orch(
    tiny_graph: tuple[SequentialOrchestrator, SequentialState, KeyArray],
) -> None:
    orch, _, _ = tiny_graph
    # Use inexact arrays (floats) as the dynamic set to avoid booleans etc.
    dyn, stat = eqx.partition(orch, eqx.is_inexact_array)
    recombined = eqx.combine(dyn, stat)
    _assert_same_treedef(orch, recombined)


def test_eqx_partition_roundtrip_state(
    tiny_graph: tuple[SequentialOrchestrator, SequentialState, KeyArray],
) -> None:
    _, state, _ = tiny_graph
    dyn, stat = eqx.partition(state, eqx.is_inexact_array)
    recombined = eqx.combine(dyn, stat)
    _assert_same_treedef(state, recombined)


def test_step_shapes_and_determinism(
    tiny_graph: tuple[SequentialOrchestrator, SequentialState, KeyArray],
) -> None:
    orch, state, key = tiny_graph
    out_state, out_key = orch.step(state, rng=key)

    assert len(out_state) == len(state)
    for i in range(len(state)):
        assert out_state[i].shape == state[i].shape
    assert isinstance(out_key, jnp.ndarray) and out_key.shape == (2,)

    # Deterministic for same inputs/seed
    out_state2, out_key2 = orch.step(state, rng=key)
    _assert_state_equal(out_state, out_state2)
    assert jnp.allclose(out_key, out_key2)


@pytest.mark.parametrize("method_name", ["step", "step_inference"])
def test_jit_matches_eager_for_each_step(
    tiny_graph: tuple[SequentialOrchestrator, SequentialState, KeyArray],
    method_name: str,
) -> None:
    """For every available step method, JIT and eager must agree exactly."""
    orch, state, key = tiny_graph
    if not hasattr(orch, method_name):
        pytest.skip(f"{method_name} not implemented")
    step_fn = getattr(orch, method_name)

    # Eager
    out_state_e, out_key_e = step_fn(state, rng=key)

    # JIT via jax.jit
    jitted = jax.jit(lambda s, k: step_fn(s, rng=k))
    out_state_j, out_key_j = jitted(state, key)

    _assert_state_equal(out_state_e, out_state_j)
    assert jnp.allclose(out_key_e, out_key_j)

    # JIT via eqx.filter_jit with orch captured statically
    @eqx.filter_jit
    def run(s: SequentialState, k: KeyArray):
        return step_fn(s, rng=k)

    out_state_f, out_key_f = run(state, key)
    _assert_state_equal(out_state_e, out_state_f)
    assert jnp.allclose(out_key_e, out_key_f)


def test_step_inference_filters_right_messages_if_present(
    tiny_graph: tuple[SequentialOrchestrator, SequentialState, KeyArray],
) -> None:
    orch, state, key = tiny_graph
    # Force visible effect of filtering by making sender 0 quite different.
    s0 = jnp.full_like(state[0], 3.0)
    s1 = jnp.full_like(state[-1], -2.0)
    state = state.replace_val(0, s0).replace_val(-1, s1)
    out_step, _ = orch.step(state, rng=key)
    out_inf, _ = orch.step_inference(state, rng=key)
    # Expect at least one receiver to change due to dropped edges.
    diffs = [jnp.sum(jnp.abs(out_step[i] - out_inf[i])) for i in range(len(state))]
    assert any(d > 0 for d in diffs), f"{out_step=}, {out_inf=}"


def test_backward_structure_and_partitionability(
    tiny_graph: tuple[SequentialOrchestrator, SequentialState, KeyArray],
) -> None:
    """Updates must mirror LayerMap structure and be partitionable for Optax."""
    orch, state, key = tiny_graph
    updates = orch.backward(state, rng=key)
    assert isinstance(updates, SequentialOrchestrator)

    # Structural keys mirror lmap
    lm = orch.lmap.to_dict()
    up = updates.lmap.to_dict()
    assert set(lm.keys()) == set(up.keys())
    for i in lm:
        assert set(lm[i].keys()) == set(up[i].keys())
        for j in lm[i]:
            # Update module type mirrors the parameter module type
            assert type(up[i][j]) is type(lm[i][j])

    # Arrays-only partition succeeds and matches tree structure
    upd_arrays, _ = eqx.partition(updates, eqx.is_inexact_array)
    params_arrays, _ = eqx.partition(orch, eqx.is_inexact_array)
    _assert_same_treedef(upd_arrays, params_arrays)


def test_layer_map_roundtrip(
    tiny_graph: tuple[SequentialOrchestrator, SequentialState, KeyArray],
) -> None:
    orch, _, _ = tiny_graph
    lmap = orch.lmap
    as_dict = lmap.to_dict()
    lmap2 = LayerMap.from_dict(as_dict, require_diagonal=True)
    assert set(as_dict.keys()) == set(lmap2.to_dict().keys())
    for i in as_dict:
        assert set(as_dict[i].keys()) == set(lmap2.to_dict()[i].keys())
