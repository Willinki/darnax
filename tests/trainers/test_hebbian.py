# tests/test_dynamical_trainer.py
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from darnax.layer_maps.sparse import LayerMap
from darnax.modules.debug import DebugAdapter, DebugLayer
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.hebbian_contrastive import ContrastiveHebbianTrainer


# -----------------------------
# Fixtures / builders
# -----------------------------
@pytest.fixture
def key() -> jax.Array:
    """Return key fixture."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def tiny_graph(key: jax.Array) -> tuple[SequentialOrchestrator, SequentialState, jax.Array]:
    """4-node graph with diagonal + some off-diagonal Debug adapters."""
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

    # sizes correspond to each node's feature size; last node size=3 → readout dim 3
    state = SequentialState(sizes=(3, 3, 3, 3))
    orch = SequentialOrchestrator(layers=lmap)
    return orch, state, key


@pytest.fixture
def trainer(
    tiny_graph: tuple[SequentialOrchestrator, SequentialState, jax.Array],
) -> ContrastiveHebbianTrainer:
    """Trainer with real orchestrator/state and a tiny Optax optimizer."""
    orchestrator, state, _ = tiny_graph
    optimizer = optax.sgd(learning_rate=0.1)
    opt_state = optimizer.init(eqx.filter(orchestrator, eqx.is_inexact_array))
    return ContrastiveHebbianTrainer(
        orchestrator=orchestrator,
        state=state,
        optimizer=optimizer,
        optimizer_state=opt_state,
        train_clamped_n_iter=1,
        train_free_n_iter=1,
        eval_n_iter=1,
    )


# -----------------------------
# Tests
# -----------------------------
def test_train_step_runs(trainer: ContrastiveHebbianTrainer, key: jax.Array) -> None:
    """Ensure train_step runs and updates context."""
    rng = key
    # Match readout dimension 3 (from last node size)
    x = jnp.ones((2, 3))
    y = jnp.ones((2, 3))

    rng = trainer.train_step(x, y, rng)

    assert isinstance(rng, jax.Array)
    assert "optimizer_state" in trainer.ctx
    assert isinstance(trainer.orchestrator, SequentialOrchestrator)
    assert isinstance(trainer.state, SequentialState)


def test_eval_step_runs(trainer: ContrastiveHebbianTrainer, key: jax.Array) -> None:
    """Ensure eval_step runs and returns metrics dict."""
    rng = key
    x = jnp.ones((2, 3))
    y = jnp.ones((2, 3))

    rng, metrics = trainer.eval_step(x, y, rng)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert metrics["accuracy"].shape == ()  # scalar array


def test_ctx_validation(trainer: ContrastiveHebbianTrainer) -> None:
    """Ensure validate_ctx enforces required keys."""
    good_ctx = trainer.validate_ctx(trainer.ctx)
    assert all(k in good_ctx for k in ("optimizer", "optimizer_state", "free_iter"))

    with pytest.raises(ValueError):
        trainer.validate_ctx({"optimizer": optax.sgd(0.1)})  # type: ignore[arg-type]


def test_jit_cache_reuse(trainer: ContrastiveHebbianTrainer, key: jax.Array) -> None:
    """Ensure that repeated train/eval calls reuse JITted functions."""
    rng = key
    x = jnp.ones((1, 3))
    y = jnp.ones((1, 3))

    # First call compiles
    trainer.train_step(x, y, rng=rng)
    first_jit = trainer._jit_train

    # Second call reuses cached compiled function
    trainer.train_step(x, y, rng=rng)
    assert trainer._jit_train is first_jit
