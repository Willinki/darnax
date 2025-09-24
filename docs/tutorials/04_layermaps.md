# 04 — LayerMaps

A **LayerMap** is a PyTree wrapper that organizes the full set of **layers** and **adapters** in your network. It provides a consistent way to index them, guarantees immutability of the structure, and integrates seamlessly with JAX/Equinox/Optax.

---

## 1. Matrix view

A LayerMap is conceptually a square matrix indexed by layer IDs:

* **Diagonal `(i, i)`**: the *i-th layer*. Each layer is stateful and sends a message to itself (its recurrent/self term).
* **Off-diagonal `(i, j)` with `i ≠ j`**: an *adapter*. It converts the j-th state into a message for the i-th layer.

This gives the following interpretation:

* **Lower triangle (`i > j`)**: forward adapters, messages flowing left → right.
* **Upper triangle (`i < j`)**: backward adapters, messages flowing right → left.
* **Row `j`**: everything that contributes **into** layer `j`.
* **Column `i`**: everything that originates **from** the state of layer `i`.
* **Row 0**: all connections going from layers to the input (unused for now).
* **Row L**: all connections going from layers to the output, meaning every layer whose state is used for prediction.
* **Column 0**: Forward skip connections from the input to each layer.
* **Column L**: Backward skip connections from the output to each layer.

This matches the convention in `SequentialState`: `s[0]` is the input state, and `s[L]` is the output state.

This structure, being in the end a dictionary of dictionaries is well suited for sparsity. For example, if layer `j` is not connected
to layer `i`, there is no adapter in the `(i, j)` position. The element is simply not present in the structure, only the relevant modules
are present.

---

## 2. API overview

```python
lm = LayerMap.from_dict({...})

lm[i]: dict                        # read-only mapping of neighbors for row i
lm[i, j]: Module                   # single module at (i, j)
(i, j) in lm: bool                 # check if edge exists

lm.rows(): tuple[int, ...]         # all row indices
lm.cols_of(i): tuple[int, ...]     # all column indices in row i
lm.neighbors(i): dict[int, Module] # read-only mapping {j: module} for row i

lm.row_items(): Iterator[int, Module]                # iterate (row, neighbors)
lm.edge_items(): Iterator[tuple[int , int], Module]  # iterate ((i, j), module)
lm.to_dict(): dict[int, dict[int, Module]]           # copy as a dict-of-dicts
```

The API is intentionally dict-like but read-only: once built, the **structure cannot be mutated**. You cannot add layers or edges later.

---

## 3. Immutability of structure

A LayerMap is **frozen** once created:

* Row/column indices (the “shape” of the map) are part of the static treedef.
* Modules on each edge can change their parameters (via updates), but the adjacency cannot change.

This immutability is not arbitrary. It is a **basic requirement in JAX**:

* The shape and structure of PyTrees must be static across JIT-compiled functions.
* If you were allowed to add or remove layers after creation, JIT cache keys would break and the compiled computation graph would need to be rebuilt every time.
* By freezing the structure, we ensure stability of compiled functions and allow the optimizer (Optax) to work on the entire network consistently.

Thus, in JAX, **data changes are dynamic**, but **structure is static**.

---

## 4. Example: building a simple LayerMap

```python
import jax
from darnax.modules.recurrent import RecurrentDiscrete
from darnax.modules.adapters import Ferromagnetic
from darnax.layer_maps.sparse import LayerMap

key0, key1 = jax.random.split(jax.random.PRNGKey(0))
F = 8

# Define two layers
layer0 = RecurrentDiscrete(features=F, j_d=0.0, threshold=0.0, key=key0)
layer1 = RecurrentDiscrete(features=F, j_d=0.0, threshold=0.0, key=key1)

# Define adapters
fwd_10 = Ferromagnetic(features=F, strength=0.5)  # forward (0 -> 1)
bwd_01 = Ferromagnetic(features=F, strength=0.2)  # backward (1 -> 0)

# Build dict-of-dicts
raw = {
    0: {0: layer0, 1: bwd_01},
    1: {0: fwd_10, 1: layer1},
}

lm = LayerMap.from_dict(raw, require_diagonal=True)

# Access
print(lm[1, 1])   # layer1
print(lm[1, 0])   # forward adapter
print(lm[1].keys())  # neighbors of row 1: {0, 1}
```

---

## 5. A LayerMap as a PyTree

Because `LayerMap` is registered as a PyTree:

* The **keys** (rows, columns) are static.
* The **modules** (layers/adapters) are leaves.
* Arrays inside those modules are visible to JAX and Optax.

This means you can treat the **entire network** as a single object:

```python
import equinox as eqx, optax

opt = optax.adam(1e-2)
opt_state = opt.init(eqx.filter(lm, eqx.is_inexact_array))

# Later in training
updates, opt_state = opt.update(grads, opt_state, params=lm)
lm = eqx.apply_updates(lm, updates)
```

All parameters inside all layers/adapters are updated in one go.

---

## 6. Summary

* **LayerMap** = a collection of layers (diagonal) and adapters (off-diagonal) with integer keys.
* **Matrix view**: rows = inputs to a layer, columns = outputs from a layer.
* **Input/output rows and columns** handle special roles.
* **Immutable structure**: you cannot add or remove layers once built. This ensures **JAX stability** (PyTree structure must be static under JIT).
* **PyTree integration**: treat the whole network as one object, pass it to Equinox/Optax, and every parameter is handled correctly.

This design makes LayerMap a central abstraction: a **static graph of modules** whose parameters evolve dynamically during training, while its topology remains fixed.

---

## 7. An ascii art

```
LayerMap (rows = receivers, columns = senders)

            columns (senders: states/messages from j) →
            0          1          2        ...        L-1         L
         ┌────────  ────────  ────────   ────────   ────────   ────────┐
r    0   │[L00]     [A01]     [A02↑]   …            [A0,L-1]   [A0L↑]  │
o        │layer0    back      back                  back       back    │
w        │(input)   adapters  adapters              adapters   adapters│
s        ├─────────────────────────────────────────────────────────────┤
(    1   │[A10↓]    [L11]     [A12↑]   …            [A1,L-1]   [A1L]   │
r        │fwd→      layer1    ↑back                 back       back    │
e        │adapters            adapters              adapters   adapters│
c        ├─────────────────────────────────────────────────────────────┤
e    2   │[A20↓]    [A21↓]    [L22]    …            [A2,L-1↑]  [A2L↑]  │
i        │fwd→      fwd→      layer2                ↑ back     ↑ back  │
v        │adapters  adapters                        adapters   adapters│
e        ├─────────────────────────────────────────────────────────────┤
r    …   │…          …        …        …            …          …       │
s        ├─────────────────────────────────────────────────────────────┤
     L   │[AL0↓]    [AL1↓]    [AL2↓]   …            [AL,L-1↓]  [LL]    │
         │fwd→      fwd→      fwd→                  fwd→       ↑layerL │
         │adapters  adapters  adapters              adapters   (output)│
         └─────────────────────────────────────────────────────────────┘
                         ↑
           rows (receivers: layer i to be updated)
```

Legend:

- Lii   : layer on the diagonal (stateful). L00 is the input-layer slot; LL is the output-layer slot.
- Aij↓  : adapter at (i,j) with i > j (lower triangle) — forward message (from j → i).
- Aij↑  : adapter at (i,j) with i < j (upper triangle) — backward message (from j → i).

Row/Column intuition:

- Row i collects everything needed to update layer i: the diagonal Lii (self-message) plus all Aij that transform state j into a message for i.
- Column j lists everything that uses state j as a source: the diagonal Ljj plus all Aij that send j’s state to other layers.

Input/Output:

- First column (·,0): forward skip connections from the input state to every layer.
- Last column (·,L): backward skip connections from the output state to earlier layers.
- Last row (L,·): all contributors that feed directly into the output layer (final prediction).

Structure:

- Diagonal = layers; off-diagonal = adapters.
- The LayerMap’s structure (rows/cols and which edges exist) is immutable after creation.
