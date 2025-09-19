# 05 — Orchestrators

Having introduced **states**, **modules**, and **layermaps**, we now come to the final structural element: the **orchestrator**.
Where the previous abstractions describe *what the network is*, the orchestrator specifies *how the network evolves in time*.
It is the execution engine of the architecture: at each iteration it routes messages, applies updates, and advances the global state.

---

## 1. Conceptual role

The orchestrator governs the **dynamical process** that characterizes this family of recurrent networks.

As everything in the library, we propose a simple interface for the object, described below, and a first
implementation. In this case, we focus on the `SequentialOrchestrator`, that plays nicely with sequential states
and the sequential nature of the layer map. In detail, at each step, this specific orchestrator does the following:

1. **Message collection** — for each receiving layer $i$, all modules $(i,j)$ in the LayerMap are evaluated on the current sender states $s^{(j)}$.
2. **Aggregation and activation** — the diagonal module $(i,i)$ aggregates the incoming messages into a pre-activation $h^{(i)}$ via its `reduce` method, then applies its nonlinearity with `activation`, yielding the new state slice $s^{(i)}$.
3. **State update** — the new slice replaces the old one in the global state, producing the next global configuration.
4. **Learning updates** — in the training regime, each module also provides a **local parameter update** through its `backward(x, y, y_hat)` method. These are gathered into a LayerMap-structured PyTree of updates.

In this way, the orchestrator realizes the iterative dynamics described in the paper: a **distributed, gradient-free learning mechanism** in which information is exchanged locally and parameters are updated via local rules.

---

## 2. Two phases of dynamics

In line with the two-phase protocol introduced in the theoretical model, the orchestrator provides two distinct update functions:

* **Training phase (`step`)**
  All available messages are considered, both forward (lower triangle) and backward (upper triangle). This corresponds to the supervised or clamped regime where input and output information are both present.

* **Inference phase (`step_inference`)**
  Only causal messages are retained: for receiver $i$, senders $j < i$ are discarded. Thus, information from the output or “future” layers does not leak backward. This corresponds to the free relaxation regime in which the system stabilizes autonomously.

This explicit separation ensures that training and inference dynamics are clearly distinguished in the implementation.

---

## 3. Public API

All orchestrators subclass the following abstract interface:

```python
class AbstractOrchestrator(eqx.Module):
    lmap: LayerMap  # fixed topology (rows, columns, edges)

    def step(self, state: StateT, *, rng: KeyArray) -> tuple[StateT, KeyArray]:
        """Run one full update step (training phase)."""

    def step_inference(self, state: StateT, *, rng: KeyArray) -> tuple[StateT, KeyArray]:
        """Run one update step (inference phase, discarding rightward messages)."""

    def backward(self, state: StateT, rng: KeyArray) -> LayerMap:
        """Compute module-local updates in a LayerMap-structured PyTree."""
```

### `step(state, rng)`

* Executes one synchronous update of the network using *all* messages.
* Returns the updated state and an advanced random key.

### `step_inference(state, rng)`

* Executes one update considering only messages from $j \geq i$.
* Used for prediction after training, ensuring purely causal message passing.

### `backward(state, rng)`

* For each edge $(i,j)$, invokes `lmap[i,j].backward(x=state[j], y=state[i], y_hat=local_field)` to obtain a **module-shaped update**.
* Returns a **LayerMap** with the same static structure as the original, but whose leaves are parameter updates.
* This PyTree can be passed directly to Optax as if it were a gradient structure.

---

## 4. Structural properties

* **Static topology**: The orchestrator’s LayerMap has a fixed set of rows, columns, and edges. This immutability is necessary for **JAX compatibility**, as PyTree structures must remain constant across compiled functions.
* **Dynamic values**: Within this static skeleton, the array values of module parameters evolve freely during training.
* **PyTree compliance**: Because the orchestrator itself is an Equinox module, it is also a PyTree. Its parameters can be filtered, updated, and optimized exactly like any other object in the system.
* **Transformation compatibility**: The orchestrator is fully compatible with `jax.jit`, `jax.vmap`, and all other JAX transformations. Since the topology is static, compilation is stable; only array values trigger recompilation when their shapes change.

---

## 5. Typical usage

A typical training loop involving an orchestrator proceeds as follows:

```python
# Forward update (training regime)
state, rng = orchestrator.step(state, rng=rng)

# Forward update (inference regime)
state, rng = orchestrator.step_inference(state, rng=rng)

# Compute module-local updates
upd_lmap = orchestrator.backward(state, rng=rng)

# Apply updates with Optax
grads  = eqx.filter(upd_lmap, eqx.is_inexact_array)
params = eqx.filter(orchestrator.lmap, eqx.is_inexact_array)
deltas, opt_state = opt.update(grads, opt_state, params=params)
new_lmap = eqx.apply_updates(orchestrator.lmap, deltas)

# Replace the LayerMap inside the orchestrator
orchestrator = eqx.tree_at(lambda o: o.lmap, orchestrator, new_lmap)
```

Here the **updates are not gradients**: they are the outcome of local learning rules defined at the module level. Optax is used purely as a robust update engine.

---

## 6. More complex logic

Right until now we described a specific instance of orchestrator, i.e. the `SequentialOrchestrator`.
It plays nicely with both the sequantial state and the sequential implementation of the layer map.

The only assumption about this structures is a notion of _order_ in the network, meaning that the first layer comes first, the second comes second, etc...

This is totally arbitrary, this library allows for any structure and logic in the network functioning, a first example might be a totally synchronous
network, where there is no notion of order and all layers are treated in sync. It is also possible to define a _group_ structure, where different layers
belong to different groups, each handled concurrently. This might involve an extension of the `LayerMap` to allow for string keys to identify groups...

Another option is to specialize architectures for speed and efficiency. Instead of working with dict of dicts and simple loops, we might want to decide
to pad and stack layer states together to be handled in parallel. This is also an easy extension of `LayerMap` and `Orchestrator`.

We might even put orchestrators inside single modules, to encapsulate a into a single object complex logic.

## 7. Summary

* The **orchestrator** advances the network’s dynamics by routing messages, aggregating them, and updating the global state.
* **`step`** executes the full supervised/clamped update; **`step_inference`** executes the free, causal update.
* **`backward`** collects local updates into a LayerMap-shaped PyTree, aligned with the parameter structure, enabling seamless integration with Optax.
* The orchestrator is a **PyTree with static structure**: immutable topology, mutable parameter values. This guarantees full compatibility with JAX transformations and ensures efficient compilation.

Through the orchestrator, the network acquires its **temporal dimension**: states evolve, messages flow, and local rules drive learning, exactly as described in the underlying theoretical framework.
