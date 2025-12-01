# search_optuna_asha.py  (Ray-free single run)
import copy
import logging
import time
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Self, Tuple

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
import wandb
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from darnax.datasets.classification.registry import build_dataset
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.fully_connected import (
    SparseFullyConnected,
    Wback,
    Wout,
)
from darnax.modules.input_output import OutputLayer
from darnax.modules.recurrent import SparseRecurrentDiscrete
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.registry import build_trainer
from darnax.utils.typing import PyTree


def build_model(
    seed: int,
    dim_data: int,
    dim_hidden: int,
    sparsity: float,
    sparsity_win: float,
    num_labels: int,
    strength_forth: float,
    strength_back: float,
    threshold_in: float,
    threshold_out: float,
    threshold_back: float,
    threshold_j: float,
    j_d: float,
) -> tuple[SequentialState, SequentialOrchestrator]:
    """Build the fully connected baseline recurrent model."""
    state = SequentialState((dim_data, dim_hidden, num_labels))

    master_key = jax.random.key(seed)
    keys = jax.random.split(master_key, num=5)

    layer_map = {
        1: {
            0: SparseFullyConnected(
                in_features=dim_data,
                out_features=dim_hidden,
                strength=strength_forth,
                threshold=threshold_in,
                sparsity=sparsity_win,
                key=keys[0],
            ),
            1: SparseRecurrentDiscrete(
                features=dim_hidden,
                j_d=j_d,
                sparsity=sparsity,
                threshold=threshold_j,
                key=keys[1],
            ),
            2: Wback(
                in_features=num_labels,
                out_features=dim_hidden,
                strength=strength_back,
                threshold=threshold_back,
                key=keys[2],
            ),
        },
        2: {
            1: Wout(
                in_features=dim_hidden,
                out_features=num_labels,
                strength=1.0,
                threshold=threshold_out,
                key=keys[3],
            ),
            2: OutputLayer(),
        },
    }

    layer_map = LayerMap.from_dict(layer_map)
    orchestrator = SequentialOrchestrator(layers=layer_map)

    return state, orchestrator


def make_lr_map_v2(
    model,
    overrides: Mapping[Tuple[int, int], str] | None = None,
    default_label: str = "default",
) -> "PyTree":
    """
    Given an Equinox model with a field `lmap[i][j]` (matrix of submodules),
    returns a pytree of string labels suitable for optax.multi_transform.

    Args:
        model: orchestrator model instance (must have .lmap).
        overrides: mapping {(i, j): label} for any cells that should get a custom label.
                   Example: {(0, 0): "enc", (0, 1): "dec", (1, 0): "heads"}
        default_label: label name for all other parameters.

    Returns:
        A pytree matching model's parameters, with string labels.
    """
    params, _ = eqx.partition(model, eqx.is_inexact_array)

    def like(tree, value):
        # Broadcast a scalar `value` to a tree with same leaves shape/structure
        return jtu.tree_map(lambda _: value, tree, is_leaf=eqx.is_array)

    # Start with all parameters labeled as default
    labels = jtu.tree_map(lambda _: default_label, params, is_leaf=eqx.is_array)

    # Apply any (i, j) label overrides
    if overrides:
        for (i, j), label in overrides.items():
            labels = eqx.tree_at(
                lambda m: m.lmap[i][j],
                labels,
                replace=like(params.lmap[i][j], label),
            )

    return labels


def decay(
    orchestrator: SequentialOrchestrator,
    config: Dict[str, Any],
) -> Any:
    """
    Return a new orchestrator with exponentially decayed weights.

    Each weight matrix is multiplied by (1 - rho), simulating
    weight decay instead of hard normalization.

    Parameters
    ----------
    orchestrator : Any
        The orchestrator module containing submodules in lmap[i][j].
    rho : float
        Decay rate. Each weight is multiplied by (1 - rho).

    Returns
    -------
    Any
        A new orchestrator with decayed weights.
    """
    # NOTE: here, we are using config value of learning rate; however, with sparsity, this is not the actual lr used.
    # This is not a bug, but it makes interpreting the weight decay harder...
    # NOTE: compared with the old codebase, we are not scaling the weight decay coefficient by the magnitude of the weights at init,
    # again because we read from config (e.g. strength_back, strength_in).
    new_orch = orchestrator

    # W_in
    # NOTE: with sparsity, here we also have a mask like for J. However, zero entries remain zero after decay, so we can skip it.
    W_in = jnp.asarray(new_orch.lmap[1][0].W)
    rescaling_win = (
        config["optimizer"]["weight_decay_win"]
        * config["optimizer"]["learning_rate_win"]
        / (config["model"]["kwargs"]["dim_data"] ** 0.5)
    )
    W_in_new = W_in * (1.0 - rescaling_win)
    new_orch = eqx.tree_at(lambda o: o.lmap[1][0].W, new_orch, W_in_new)

    # J
    J = jnp.asarray(new_orch.lmap[1][1].J)
    mask = jnp.asarray(new_orch.lmap[1][1]._mask)  # exclude diagonal from decay
    rescaling_j = (
        config["optimizer"]["weight_decay_j"]
        * config["optimizer"]["learning_rate_j"]
        / (config["model"]["kwargs"]["dim_hidden"] ** 0.5)
    )
    J_new = J * (1.0 - rescaling_j * mask)
    new_orch = eqx.tree_at(lambda o: o.lmap[1][1].J, new_orch, J_new)

    # W_out
    W_out = jnp.asarray(new_orch.lmap[2][1].W)
    rescaling_wout = (
        config["optimizer"]["weight_decay_wout"]
        * config["optimizer"]["learning_rate_wout"]
        / (config["model"]["kwargs"]["dim_hidden"] ** 0.5)
    )
    W_out_new = W_out * (1.0 - rescaling_wout)
    new_orch = eqx.tree_at(lambda o: o.lmap[2][1].W, new_orch, W_out_new)

    return new_orch


#
# DEBUG METRIC 1: CLASSES OF MISCLASSIFIED EXAMPLES
# logged as histogram
#
def misclf_hist_per_batch(batch_id, x, y, orchestrator, state) -> jax.Array:
    """Return bincount of true classes among misclassified examples in this batch."""
    y_true = jnp.argmax(y, axis=-1)
    y_predicted = jnp.argmax(state[-1], axis=-1)
    wrong_mask = y_true != y_predicted
    return y_true[wrong_mask]


def misclf_hist_aggregate(values):
    """Aggregate per-batch misclassification histograms and return a W&B histogram."""
    if not values:
        return wandb.Histogram([])
    values = jnp.concat(values)
    # Use raw counts directly (W&B handles count-based input)
    return wandb.Histogram(values)


#
# DEBUG METRIC 2: AVERAGE INTRA-CLASS - INTERCLASS overlap also as heatmap
# logged as a number


def return_internal_states(batch_id, x, y, orchestrator, state):
    if batch_id % 10 != 0:
        return None

    return (state[-2], jnp.argmax(y, axis=-1))


def compute_internal_overlap(values):
    """
    values: list of Optional[Tuple[state[B,S], labels[B]]]
      - state is binary {0,1} or bool; labels are int class ids.
    Computes:
      ratio = mean_overlap_same_class / mean_overlap_diff_class
    Overlap uses Ising-style definition:
      Convert {0,1} -> {-1,1} via s = 2*state - 1,
      q(i,j) = (1/S) * sum_k s_i[k] * s_j[k]
    Returns a Python float (loggable in W&B).
    """
    # Filter out None entries
    tuples = [t for t in values if t is not None]
    if len(tuples) == 0:
        return float("nan")

    # Concatenate across batches
    states_list, labels_list = zip(*tuples)
    Xpm = jnp.concatenate([jnp.asarray(s) for s in states_list], axis=0)  # [N, S]
    y = jnp.concatenate([jnp.asarray(l) for l in labels_list], axis=0)  # [N]
    S = Xpm.shape[1]

    # Pairwise overlaps (Gram) q_ij = (1/S) * Xpm_i @ Xpm_j
    gram = (Xpm @ Xpm.T) / float(S)  # [N, N]

    # Build masks for same/different class pairs (upper triangle, no diagonal)
    N = y.shape[0]
    yy = y[:, None]
    same = yy == yy.T
    upper = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    same_mask = same & upper
    diff_mask = (~same) & upper

    # Gather values
    same_vals = gram[same_mask]
    diff_vals = gram[diff_mask]

    # Means with empty-guard
    same_mean = jnp.where(same_vals.size > 0, jnp.mean(same_vals), jnp.nan)
    diff_mean = jnp.where(diff_vals.size > 0, jnp.mean(diff_vals), jnp.nan)

    return {
        "same_class": same_mean,
        "different_class": diff_mean,
        "diff_same_ratio": diff_mean / same_mean,
    }


def compute_internal_overlap_heatmap(values):
    """
    values: list of Optional[Tuple[state[B,S], labels[B]]]
      - state is already in {-1,+1}; labels are int class ids.

    Produces a heatmap of pairwise overlaps q(i,j) = (1/S) * sum_k s_i[k] * s_j[k],
    grouping samples by class to show block structure.
    """
    tuples = [t for t in values if t is not None]
    if not tuples:
        return None

    # concatenate
    states_list, labels_list = zip(*tuples)
    Xpm = jnp.concatenate(states_list, axis=0)  # [N, S]
    y = jnp.concatenate(labels_list, axis=0)  # [N]

    # compute pairwise overlaps
    S = Xpm.shape[1]
    gram = (Xpm @ Xpm.T) / float(S)  # [N,N]

    # reorder by class
    classes = np.unique(np.array(y))
    order = np.concatenate([np.where(np.array(y) == c)[0] for c in classes])
    gram_ord = gram[order][:, order]
    y_ord = np.array(y)[order]

    # compute block boundaries
    boundaries = []
    start = 0
    for c in classes:
        n = np.sum(y_ord == c)
        boundaries.append((start, start + n))
        start += n

    # plot heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.array(gram_ord), vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")

    for s, e in boundaries:
        if s > 0:
            ax.axhline(s - 0.5, color="k", lw=1)
            ax.axvline(s - 0.5, color="k", lw=1)

    ax.set_title("Internal State Overlaps (grouped by class)")
    ax.set_xlabel("samples (grouped by class)")
    ax.set_ylabel("samples (grouped by class)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="q(i,j)")
    return fig


#
# DEBUG METRIC 3: weights
# logged as a histogram
def get_weights(batch_id, x, y, orchestrator, state):
    if batch_id != 0:
        return None

    return {
        "J": wandb.Histogram(orchestrator.lmap[1][1].J),
        "W_in": wandb.Histogram(orchestrator.lmap[1][0].W),
        "W_out": wandb.Histogram(orchestrator.lmap[2][1].W),
    }


def pass_weights(values):
    values = [x for x in values if x is not None]
    assert len(values) == 1
    return values[0]


#
# DEBUG METRIC 4: INTERNAL FIELDS
#
def get_fields(batch_id, x, y, orchestrator, state):
    # no effect
    state = state.replace_val(-1, y)
    sub = jax.random.key(seed=44)
    left_field = orchestrator.lmap[1][0](state[0])
    right_field = orchestrator.lmap[1][2](state[2])
    self_field = orchestrator.lmap[1][1](state[1])
    return (left_field, self_field, right_field)


def summarize_fields(values):
    left_fields = jnp.concat(list(map(lambda x: x[0], values)))
    self_fields = jnp.concat(list(map(lambda x: x[1], values)))
    right_fields = jnp.concat(list(map(lambda x: x[2], values)))
    return {
        "left_fields": wandb.Histogram(left_fields),
        "right_fields": wandb.Histogram(right_fields),
        "self_fields": wandb.Histogram(self_fields),
    }


#
# DEBUG METRIC 4: LABEL FIELDS
#
def get_label(batch_id, x, y, orchestrator, state):
    return (x.flatten(), y.flatten())


def summarize_labels(values):
    x = jnp.concat(list(map(lambda x: x[0], values)))
    y = jnp.concat(list(map(lambda x: x[1], values)))
    return {
        "x": wandb.Histogram(x),
        "y": wandb.Histogram(y),
    }


#
# DEBUG METRIC 5: LABEL FIELDS
#
def get_label(batch_id, x, y, orchestrator, state):
    return state[-2]


def summarize_states(values):
    state = jnp.concat(values)
    return state.mean()


DEBUG_METRICS = {
    "error_class": (misclf_hist_per_batch, misclf_hist_aggregate),
    "overlap_states": (return_internal_states, compute_internal_overlap),
    "overlap_figures": (return_internal_states, compute_internal_overlap_heatmap),
    "weights": (get_weights, pass_weights),
    "fields": (get_fields, summarize_fields),
    "data": (get_label, summarize_labels),
    "final_state": (get_label, summarize_states),
}


JaxArray = jax.Array


def init_debug_buckets(debug_metrics: Dict[str, Tuple[callable, callable]]):
    return {name: [] for name in debug_metrics.keys()}


def update_debug_buckets(
    buckets: Dict[str, List[JaxArray]],
    debug_metrics: Dict[str, Tuple[callable, callable]],
    batch_id: int,
    x: jax.Array,
    y: jax.Array,
    orchestrator: object,
    state: object,
):
    for name, (per_batch, _) in debug_metrics.items():
        buckets[name].append(per_batch(batch_id, x, y, orchestrator, state))


def aggregate_debug_buckets(
    buckets: Dict[str, List[JaxArray]],
    debug_metrics: Dict[str, Tuple[callable, callable]],
):
    aggregated = {}
    for name, (_, aggregate) in debug_metrics.items():
        aggregated[name] = aggregate(buckets[name])
    return aggregated


def flatten_for_logging(prefix: str, aggregated: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for k, v in aggregated.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{prefix}{k}/{kk}"] = vv
        else:
            out[f"{prefix}{k}"] = v
    return out


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="ours_emnist")
def train_once(cfg) -> None:
    print("beginning...")
    cfg = deepcopy(cfg)
    key = jax.random.key(cfg.get("master_seed", 0))
    wb = cfg["wandb"]
    if wb.get("enabled", True):
        cfg_for_wandb = OmegaConf.to_container(cfg, resolve=True)

        wandb.init(
            entity=wb["entity"],
            project=wb["project"],
            name=wb["run_name"],
            mode=wb["mode"],
            dir=wb["dir"],
            config=cfg_for_wandb,
            tags=list(wb.get("tags", [])),
            save_code=wb.get("save_code", True),
        )

    print("initialized wandb")
    state, orchestrator = build_model(**cfg["model"]["kwargs"])
    ds = build_dataset(cfg["data"]["name"], **cfg["data"]["kwargs"])
    key, data_key = jax.random.split(key)
    ds.build(data_key)

    lr_map = make_lr_map_v2(
        orchestrator,
        overrides={(1, 0): "w_in", (1, 1): "j", (2, 1): "w_out"},
    )
    lr_win = cfg["optimizer"]["learning_rate_win"]
    lr_wout = cfg["optimizer"]["learning_rate_wout"]
    lr_j = cfg["optimizer"]["learning_rate_j"]

    print("Rescaling learning rates to account for sparsity...")
    lr_j /= jnp.sqrt(1 - cfg.model.kwargs.sparsity)
    lr_win /= jnp.sqrt(1 - cfg.model.kwargs.sparsity_win)

    optimizer = optax.multi_transform(
        {
            "default": optax.sgd(learning_rate=0.0),
            "w_in": optax.sgd(learning_rate=lr_win),
            "w_out": optax.sgd(learning_rate=lr_wout),
            "j": optax.sgd(learning_rate=lr_j),
        },
        lr_map,
    )
    opt_state = optimizer.init(eqx.filter(orchestrator, eqx.is_inexact_array))

    trainer = build_trainer(
        cfg["trainer"]["name"],
        orchestrator=orchestrator,
        state=state,
        optimizer=optimizer,
        optimizer_state=opt_state,
        **cfg["trainer"]["kwargs"],
    )

    print("initialized trainer")

    # ---- saving norms ----
    for epoch in range(0, int(cfg["epochs"]) + 1):
        t0 = time.time()

        # ---- Train ----
        if epoch != 0:
            # count = 0
            # avg_logs = defaultdict(float)
            for xb, yb in ds:
                use_gating = cfg.trainer.gating.enabled and (
                    epoch > cfg.trainer.gating.warmup_epochs
                )
                key, logs = trainer.train_step(
                    xb,
                    yb,
                    key,
                    use_gating=use_gating,
                    gating_shift=cfg.trainer.gating.shift,
                    fake_dynamics=cfg.trainer.fake_dynamics.enabled,
                    fake_dynamics_k=cfg.trainer.fake_dynamics.k,
                    fake_dynamics_vanilla=cfg.trainer.fake_dynamics.vanilla,
                    double_dynamics=cfg.trainer.double_dynamics,
                )
                trainer.orchestrator = decay(trainer.orchestrator, cfg)

                # count += 1
                # for k, v in logs.items():
                #     avg_logs[k] += v
            # for k in avg_logs:
            #     avg_logs[k] /= count
            # if wb.get("enabled", True):
            #     wandb.log(avg_logs, step=epoch, commit=False)

        # ---- Eval (test) + per-batch debug ----
        accs_eval = []
        buckets = init_debug_buckets(DEBUG_METRICS)
        for b_index, (xb, yb) in enumerate(ds.iter_test()):
            key, metrics = trainer.eval_step(xb, yb, key)
            accs_eval.append(metrics["accuracy"])
            update_debug_buckets(
                buckets=buckets,
                debug_metrics=DEBUG_METRICS,
                batch_id=b_index,
                x=xb,
                y=yb,
                orchestrator=trainer.orchestrator,
                state=trainer.state,
            )

        acc_eval = float(jnp.mean(jnp.array(accs_eval))) if accs_eval else float("nan")
        aggregated_debug = aggregate_debug_buckets(buckets, DEBUG_METRICS)
        debug_log = flatten_for_logging(prefix="debug/", aggregated=aggregated_debug)

        # ---- Eval (train split) for train accuracy (unchanged) ----
        accs_train = []
        for b_index, (xb, yb) in enumerate(ds):
            key, metrics = trainer.eval_step(xb, yb, key)
            accs_train.append(metrics["accuracy"])
        acc_train = float(jnp.mean(jnp.array(accs_train))) if accs_train else float("nan")

        if wb.get("enabled", True):
            log_content = debug_log
            if epoch != 0:
                log_content |= {
                    "accuracy/train": acc_train,
                    "accuracy/eval": acc_eval,
                    "epoch_time_s": time.time() - t0,
                }
            wandb.log(
                log_content,
                step=epoch,
                commit=True,
            )

        print(
            f"Epoch {epoch:03d} | train_acc={acc_train:.4f} | eval_acc={acc_eval:.4f} | time={time.time() - t0:.2f}s"
        )

    # ---- Optional PyTorch linear classifier on learned representations ----
    if "torch_clf" in cfg and cfg.torch_clf.get("enabled", False):
        torch_clf_cfg = cfg.torch_clf
        print("Starting PyTorch linear classifier training...")
        # Seed
        if "master_seed" in cfg:
            torch.manual_seed(int(cfg.master_seed))

        # Feature extraction: TRAIN split
        train_reps = []
        train_labels = []
        for xb, yb in ds:  # train split
            key, _ = trainer.eval_step(xb, yb, key)
            reps = trainer.state.representations
            train_reps.append(copy.deepcopy(np.array(reps)))
            yb_np = np.array(copy.deepcopy(yb))
            yb_np = np.argmax(yb_np, axis=-1)
            train_labels.append(copy.deepcopy(yb_np))

        # Feature extraction: TEST/EVAL split
        test_reps = []
        test_labels = []
        for xb, yb in ds.iter_test():
            key, _ = trainer.eval_step(xb, yb, key)
            reps = trainer.state.representations
            test_reps.append(copy.deepcopy(np.array(reps)))
            yb_np = np.array(copy.deepcopy(yb))
            yb_np = np.argmax(yb_np, axis=-1)
            test_labels.append(copy.deepcopy(yb_np))

        features_train = torch.from_numpy(np.concatenate(train_reps, axis=0)).float()
        labels_train = torch.from_numpy(np.concatenate(train_labels, axis=0)).long()
        features_test = torch.from_numpy(np.concatenate(test_reps, axis=0)).float()
        labels_test = torch.from_numpy(np.concatenate(test_labels, axis=0)).long()

        input_dim = int(features_train.shape[1])
        num_classes = int(cfg.model.kwargs.num_labels)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Linear(input_dim, num_classes, bias=cfg.torch_clf.get("use_bias", False)).to(
            device
        )
        criterion = nn.CrossEntropyLoss()
        opt = torch_clf_cfg.get("optimizer", "adam").lower()
        opt_class = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[opt]
        optimizer = opt_class(
            model.parameters(),
            lr=float(torch_clf_cfg.lr),
            weight_decay=float(torch_clf_cfg.weight_decay),
        )

        batch_size = int(torch_clf_cfg.batch_size)
        train_loader = DataLoader(
            TensorDataset(features_train, labels_train),
            batch_size=batch_size,
            shuffle=True,
        )

        prefix = str(torch_clf_cfg.get("log_prefix", "torch_clf"))
        epochs_clf = int(torch_clf_cfg.epochs)
        base_step = int(cfg["epochs"]) + 1  # start after last JAX epoch index

        for e in range(epochs_clf):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for xb_t, yb_t in train_loader:
                xb_t = xb_t.to(device)
                yb_t = yb_t.to(device)
                optimizer.zero_grad()
                logits = model(xb_t)
                loss = criterion(logits, yb_t)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * yb_t.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == yb_t).sum().item()
                total += yb_t.size(0)
            train_loss = total_loss / max(1, total)
            train_acc = correct / max(1, total)

            model.eval()
            with torch.no_grad():
                logits_eval = model(features_test.to(device))
                eval_loss = criterion(logits_eval, labels_test.to(device)).item()
                pred_eval = logits_eval.argmax(dim=1)
                eval_acc = (pred_eval == labels_test.to(device)).float().mean().item()

            if cfg.wandb.get("enabled", True):
                wandb.log(
                    {
                        f"{prefix}/train_loss": train_loss,
                        f"{prefix}/train_acc": train_acc,
                        f"{prefix}/eval_loss": eval_loss,
                        f"{prefix}/eval_acc": eval_acc,
                        f"{prefix}/epoch": e,
                    },
                    step=base_step + e,
                    commit=True,
                )

            print(
                f"[Torch Clf] Epoch {e:03d} | train_acc={train_acc:.4f} | eval_acc={eval_acc:.4f} | train_loss={train_loss:.4f} | eval_loss={eval_loss:.4f}"
            )

        print("PyTorch linear classifier training complete.")

    if wb.get("enabled", True):
        wandb.finish()


if __name__ == "__main__":
    train_once()
