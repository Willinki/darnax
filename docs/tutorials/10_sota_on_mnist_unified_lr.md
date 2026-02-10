---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: py:percent,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
---

# 10 — Sparsity and module-internal learning rate and weight decay.

This notebook-as-a-script trains a sparse recurrent model with darnax on MNIST.

We show how to:

- build a sparse recurrent architecture,
- pass learning rate and weight decay directly to modules,
- train a separate torch classifier on the final representations.


```python
import copy
from copy import deepcopy
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from darnax.datasets.classification.mnist import Mnist
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.fully_connected import (
    FrozenRescaledFullyConnected,
    FullyConnected,
    SparseFullyConnected,
)
from darnax.modules.input_output import OutputLayer
from darnax.modules.recurrent import SparseRecurrentDiscrete
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer
```

## 2. Experiment configuration (`PARAMS`)

We provide config in a simple Python dictionary.

```python
PARAMS: dict[str, Any] = {
    "master_seed": 0,
    "epochs": 20,
    "model": {
        "kwargs": {
            "seed": 44,
            "dim_data": 784,  # input dimension
            "dim_hidden": 2000,  # hidden size
            "num_labels": 10,  # number of classes
            "sparsity": 0.99,  # recurrent sparsity
            "sparsity_win": 0.9,  # input sparsity
            "strength_forth": 5.0,
            "strength_back": 1.3,
            "j_d": 0.95,  # self-coupling in recurrent J
            "threshold_in": 1.78,
            "threshold_out": 7.0,
            "threshold_j": 1.78,
            "threshold_back": 0.0,
        }
    },
    "data": {
        "kwargs": {
            "batch_size": 16,
            "linear_projection": None,
            "num_images_per_class": None,
            "label_mode": "pm1",
            "x_transform": "identity",
            "flatten": True,
        },
    },
    "optimizer": {
        "learning_rate_win": 0.159,  # input layer
        "learning_rate_j": 0.058,  # recurrent J
        "learning_rate_wout": 0.17,  # output layer
        "weight_decay_win": 0.01,
        "weight_decay_j": 0.00006,
        "weight_decay_wout": 0.02,
    },
    "trainer": {
        "kwargs": {
            "warmup_n_iter": 1,
            "train_clamped_n_iter": 5,
            "train_free_n_iter": 5,
            "eval_n_iter": 5,
        },
    },
    "torch_clf": {
        "enabled": True,
        "use_bias": False,
        "optimizer": "adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 256,
        "epochs": 20,
    },
}
```

## 3. Building the sparse recurrent model

We construct a `SequentialOrchestrator` with:

- sparse input adapter (`SparseFullyConnected`),
- sparse recurrent core (`SparseRecurrentDiscrete`),
- dense, fixed, feedback
- dense output layer

```python
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
    lr_win: float,
    lr_j: float,
    lr_wout: float,
    wd_win: float,
    wd_j: float,
    wd_wout: float,
) -> tuple[SequentialState, SequentialOrchestrator]:
    """Build a sparse recurrent model with input, recurrent, output modules.

    The model has:

    - a sparse input adapter ``SparseFullyConnected``,
    - a sparse recurrent core ``SparseRecurrentDiscrete``,
    - a feedback adapter
    - an output adapter followed by ``OutputLayer``.

    Parameters
    ----------
    seed:
        PRNG seed for JAX initializations.
    dim_data:
        Input dimension.
    dim_hidden:
        Size of the recurrent layer.
    sparsity:
        Fraction of zero entries in the recurrent J matrix.
    sparsity_win:
        Fraction of zero entries in the input matrix W_in.
    num_labels:
        Number of output classes.
    strength_forth:
        Scaling factor for forward (input → hidden) couplings.
    strength_back:
        Scaling factor for feedback (output → hidden) couplings.
    threshold_in:
        Threshold for the input adapter dynamics.
    threshold_out:
        Threshold for the output adapter dynamics.
    threshold_back:
        Threshold for the feedback adapter dynamics.
    threshold_j:
        Threshold for the recurrent core.
    j_d:
        Diagonal self–coupling strength for the recurrent core.
    lr_win: Learning rate for the input weights.
    lr_j: Learning rate for the recurrent weights.
    lr_wout: Learning rate for the output weights.
    wd_win: Weight decay for the input weights.
    wd_j: Weight decay for the recurrent weights.
    wd_wout: Weight decay for the output weights.

    Returns
    -------
    state:
        Initial ``SequentialState`` with appropriate shapes.
    orchestrator:
        ``SequentialOrchestrator`` wiring all modules together.

    """
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
                lr=lr_win,
                weight_decay=wd_win,
            ),
            1: SparseRecurrentDiscrete(
                features=dim_hidden,
                j_d=j_d,
                sparsity=sparsity,
                threshold=threshold_j,
                key=keys[1],
                lr=lr_j,
                weight_decay=wd_j,
            ),
            2: FrozenRescaledFullyConnected(
                in_features=num_labels,
                out_features=dim_hidden,
                strength=strength_back,
                threshold=threshold_back,
                key=keys[2],
            ),
        },
        2: {
            1: FullyConnected(
                in_features=dim_hidden,
                out_features=num_labels,
                strength=1.0,
                threshold=threshold_out,
                key=keys[3],
                lr=lr_wout,
                weight_decay=wd_wout,
            ),
            2: OutputLayer(),
        },
    }

    layer_map = LayerMap.from_dict(layer_map)
    orchestrator = SequentialOrchestrator(layers=layer_map)

    return state, orchestrator
```

## 4. Training loop (JAX model)

We now define the main training function:

- build model, dataset, trainer,
- create a simple SGD optimizer,
- rescale learning rates by sparsity,
- train for a fixed number of epochs,
- log only train and eval accuracy (printed + returned).

```python
def train_once(params: dict[str, Any]) -> tuple[dict[str, list[float]], Any, Any, jax.Array]:
    """Train the sparse recurrent model for a fixed number of epochs.

    Parameters
    ----------
    params:
        Experiment configuration dictionary (``PARAMS``).

    Returns
    -------
    history:
        Dictionary with lists of training and evaluation accuracies.
    trainer:
        Trained trainer object.
    ds:
        Dataset object used for training and evaluation.
    key:
        Final JAX PRNG key after training.

    """
    cfg = deepcopy(params)
    key = jax.random.key(cfg.get("master_seed", 0))

    # Prepare kwargs for build_model
    model_kwargs = cfg["model"]["kwargs"]

    lr_win = cfg["optimizer"]["learning_rate_win"]
    lr_j = cfg["optimizer"]["learning_rate_j"]

    # Rescale learning rates to account for sparsity
    sparsity = cfg["model"]["kwargs"]["sparsity"]
    sparsity_win = cfg["model"]["kwargs"]["sparsity_win"]
    lr_j /= jnp.sqrt(1.0 - sparsity)
    lr_win /= jnp.sqrt(1.0 - sparsity_win)

    model_kwargs["lr_win"] = lr_win
    model_kwargs["lr_j"] = lr_j
    model_kwargs["lr_wout"] = cfg["optimizer"]["learning_rate_wout"]
    model_kwargs["wd_win"] = cfg["optimizer"]["weight_decay_win"]
    model_kwargs["wd_j"] = cfg["optimizer"]["weight_decay_j"]
    model_kwargs["wd_wout"] = cfg["optimizer"]["weight_decay_wout"]

    # Build model and dataset
    state, orchestrator = build_model(**model_kwargs)

    ds = Mnist(**cfg["data"]["kwargs"])
    key, data_key = jax.random.split(key)
    ds.build(data_key)

    # The learning rate and weight decay are now handled by the modules.
    # We use a simple SGD optimizer that just applies the updates returned
    # by the `backward` method. The `backward` method of the modules returns
    # the update with the learning rate already applied, so we use a learning
    # rate of 1.0 here.
    optimizer = optax.sgd(learning_rate=1.0)
    opt_state = optimizer.init(eqx.filter(orchestrator, eqx.is_inexact_array))

    trainer = DynamicalTrainer(
        orchestrator=orchestrator,
        state=state,
        optimizer=optimizer,
        optimizer_state=opt_state,
        **cfg["trainer"]["kwargs"],
    )

    history: dict[str, list[float]] = {
        "train_acc": [],
        "eval_acc": [],
    }

    num_epochs = int(cfg["epochs"])

    for epoch in range(0, num_epochs + 1):
        # ---- Train ----
        if epoch != 0:
            for xb, yb in ds:
                key = trainer.train_step(xb, yb, key)

        # ---- Eval on test split ----
        accs_eval = []
        for xb, yb in ds.iter_test():
            key, metrics = trainer.eval_step(xb, yb, key)
            accs_eval.append(metrics["accuracy"])

        acc_eval = float(jnp.mean(jnp.array(accs_eval))) if accs_eval else float("nan")

        # ---- Eval on train split ----
        accs_train = []
        for xb, yb in ds:
            key, metrics = trainer.eval_step(xb, yb, key)
            accs_train.append(metrics["accuracy"])
        acc_train = float(jnp.mean(jnp.array(accs_train))) if accs_train else float("nan")

        history["train_acc"].append(acc_train)
        history["eval_acc"].append(acc_eval)

        print(f"Epoch {epoch:03d} | train_acc={acc_train:.4f} | eval_acc={acc_eval:.4f}")

    return history, trainer, ds, key
```

## 5. Optional: PyTorch linear classifier on learned representations

We keep the option to train a simple linear probe on frozen JAX representations.
This is useful for evaluating representation quality.

```python
def train_torch_linear_probe(
    trainer: Any,
    ds: Any,
    cfg: dict[str, Any],
    key: jax.Array,
) -> None:
    """Train a linear classifier on frozen JAX representations using PyTorch.

    Parameters
    ----------
    trainer:
        Trained darnax trainer (provides ``.state.representations``).
    ds:
        Dataset object with train and test iterators.
    cfg:
        Experiment config with ``"model"`` and ``"torch_clf"`` sections.
    key:
        JAX PRNG key (updated during evaluation passes).

    """
    if "torch_clf" not in cfg or not cfg["torch_clf"].get("enabled", False):
        print("Torch linear probe is disabled in PARAMS['torch_clf'].")
        return

    torch_clf_cfg = cfg["torch_clf"]
    print("Starting PyTorch linear classifier training...")

    if "master_seed" in cfg:
        torch.manual_seed(int(cfg["master_seed"]))

    # Feature extraction: train split
    train_reps = []
    train_labels = []
    for xb, yb in ds:
        key, _ = trainer.eval_step(xb, yb, key)
        reps = trainer.state[-2]
        train_reps.append(copy.deepcopy(np.array(reps)))
        yb_np = np.array(copy.deepcopy(yb))
        yb_np = np.argmax(yb_np, axis=-1)
        train_labels.append(copy.deepcopy(yb_np))

    # Feature extraction: test split
    test_reps = []
    test_labels = []
    for xb, yb in ds.iter_test():
        key, _ = trainer.eval_step(xb, yb, key)
        reps = trainer.state[-2]
        test_reps.append(copy.deepcopy(np.array(reps)))
        yb_np = np.array(copy.deepcopy(yb))
        yb_np = np.argmax(yb_np, axis=-1)
        test_labels.append(copy.deepcopy(yb_np))

    features_train = torch.from_numpy(np.concatenate(train_reps, axis=0)).float()
    labels_train = torch.from_numpy(np.concatenate(train_labels, axis=0)).long()
    features_test = torch.from_numpy(np.concatenate(test_reps, axis=0)).float()
    labels_test = torch.from_numpy(np.concatenate(test_labels, axis=0)).long()

    input_dim = int(features_train.shape[1])
    num_classes = int(cfg["model"]["kwargs"]["num_labels"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(input_dim, num_classes, bias=torch_clf_cfg.get("use_bias", False)).to(device)
    criterion = nn.CrossEntropyLoss()
    opt_name = torch_clf_cfg.get("optimizer", "adam").lower()
    opt_class = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[opt_name]
    optimizer = opt_class(
        model.parameters(),
        lr=float(torch_clf_cfg["lr"]),
        weight_decay=float(torch_clf_cfg["weight_decay"]),
    )

    batch_size = int(torch_clf_cfg["batch_size"])
    train_loader = DataLoader(
        TensorDataset(features_train, labels_train),
        batch_size=batch_size,
        shuffle=True,
    )

    epochs_clf = int(torch_clf_cfg["epochs"])

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

        print(
            f"[Torch Clf] Epoch {e:03d} | "
            f"train_acc={train_acc:.4f} | eval_acc={eval_acc:.4f} | "
            f"train_loss={train_loss:.4f} | eval_loss={eval_loss:.4f}"
        )

    print("PyTorch linear classifier training complete.")
```

## 6. Run training and plot accuracies

You can execute this block to train the model and visualize
train vs. eval accuracy over epochs.

```python
if __name__ == "__main__":
    history, trainer, ds, key = train_once(PARAMS)

    plt.figure(figsize=(6, 4))
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["eval_acc"], label="eval")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Sparse recurrent model: train vs eval accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if PARAMS["torch_clf"]["enabled"]:
        train_torch_linear_probe(trainer, ds, PARAMS, key)
```
