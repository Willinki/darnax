# 02 — States

This tutorial introduces **States**: the data structures that hold the **current condition** of the network. Unlike the transient *activations* of standard feedforward networks, states here are **persistent, dynamical variables**. They evolve iteratively under the network’s dynamics and represent the system’s position in configuration space, not merely intermediate results of a forward pass.

---

## 1. Conceptual overview

A **state** records the evolving configuration of the network at each layer. In the theoretical formulation, states are denoted

\[
s^{(l)} \in \{\pm 1\}^N,
\]

but in this library they are general JAX arrays of arbitrary shape and dtype.

Key points:

* **Layer–state relation.** Every layer of the network is associated with a state buffer. This includes the **input** and **output** layers.
* **Initialization.** When a batch `(x, y)` is presented:

  * the input buffer is set to `x`,
  * the output buffer may be set to `y`,
  * all intermediate buffers are initialized to zeros.
* **Dynamics.** States evolve by iterative updates until convergence (a fixed point or a steady regime). Later tutorials will explain these dynamics in detail.
* **Shape generality.** State buffers are not restricted to vectors. They may be multi-dimensional, e.g. `(H, W, C)` for convolutional or image-like architectures. The abstraction supports arbitrary shapes per layer.

---

## 2. API responsibilities

A state is deliberately minimal in API design but aligned with JAX’s functional style. Each buffer is a JAX array, and the object provides simple functional accessors:

```python
class State(eqx.Module):
    def __getitem__(self, key: Any) -> Array: ...
    def init(self, x: Array, y: Array | None = None) -> Self: ...
    def replace(self, value: PyTree) -> Self: ...
    def replace_val(self, idx: Any, value: Array) -> Self: ...
```

* `__getitem__(key)`
  Retrieves the buffer associated with `key` (e.g., layer index).

* `init(x, y=None)`
  Functionally initializes the state for a batch. Resizes all buffers to the batch dimension of `x`, sets the input to `x`, and optionally sets the output to `y`.

* `replace(value)`
  Returns a new state with the entire collection of buffers replaced by `value`.

* `replace_val(idx, value)`
  Returns a new state with only the buffer at position `idx` replaced by `value`.

All updates are *functional* and produce new objects. This immutability is essential for compatibility with JAX transformations.

---

## 3. A concrete implementation: `SequentialState`

`SequentialState` implements a **left-to-right sequence of buffers**, indexed by integers:

* `0` = input buffer
* `1, 2, …, L-2` = intermediate buffers
* `L-1` (or `-1`) = output buffer

Internally, it stores a list of arrays with shapes `(B, *size_l)`, where `B` is the batch size.

```python
state = SequentialState(sizes=[(4,), (8,), (3,)])
```

At construction time, buffers are initialized with dummy batch size `B=1`. The `init` method resizes them to the real batch size provided by `x`.

### Example: initialization

```python
import jax
import jax.numpy as jnp
from bionet.states.sequential import SequentialState

state = SequentialState(sizes=[(4,), (8,), (3,)])

x = jax.random.normal(jax.random.PRNGKey(0), (32, 4))
y = jax.random.normal(jax.random.PRNGKey(1), (32, 3))

s0 = state.init(x, y)
assert s0[0].shape == (32, 4)  # input
assert s0[1].shape == (32, 8)  # hidden
assert s0[-1].shape == (32, 3) # output
```

---

## 4. Why a *global* state?

A deliberate design decision is to represent **the state of the entire network globally**, rather than letting each layer enclose its own state.

The reasons are:

1. **Shared storage.** Different layers may share access to portions of the same underlying state (e.g., in convolutional networks, multiple filters may act on overlapping regions of a global image-like buffer).
2. **Topological flexibility.** A global state can be organized as a single multidimensional array or structured container, with different layers responsible for reading and writing specific slices.
3. **Consistency.** This design allows heterogeneous architectures (dense, convolutional, graph-like) to operate on the same formal object without modifying the layer abstraction.

This choice reflects the dynamical perspective: the **network evolves as a whole**, not as a collection of isolated layer-local states.

---

## 5. States as pytrees

Because states are Equinox modules, they are **pytrees**. This has important consequences:

* JAX transformations (`jit`, `grad`, `vmap`, `pmap`) work seamlessly over state objects.
* Updates remain functional: replacing or modifying buffers yields new pytree instances.
* Static fields (e.g., `dtype`) are excluded from the dynamic leaves, reducing recompilation overhead.

Thus, states are simultaneously *containers of arrays* and *first-class JAX objects*.

---

## 6. Summary

* States represent **persistent network conditions**, not just transient activations.
* Every layer is associated with part of the state, including input and output.
* States can store arbitrary shapes, enabling general architectures (vector-based, convolutional, etc.).
* The API is simple: indexing, functional initialization, and functional replacement.
* The design emphasizes a **global state** abstraction, allowing layers to share and reuse portions of it.
* Being pytrees, states integrate naturally with JAX’s functional transformations.
