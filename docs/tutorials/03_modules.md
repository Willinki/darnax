# 03 — Modules (Layers and Adapters)

This tutorial introduces **Modules**, the core building blocks of the library. Modules are divided into two categories:

* **Layers**: stateful modules that read and update a slice of the global network state.
* **Adapters**: stateless modules that transform one layer’s state into a message for another.

Only **layers** own a state. Both families may carry parameters; some are trainable (producing non-zero updates), others fixed (producing zero updates).

---

## 1. Conceptual background

The library is inspired by the recurrent dynamical model described in *Dynamical Learning in Deep Asymmetric Recurrent Neural Networks*.
In that setting, each neuron (or unit) maintains a binary state $s \in \{-1, 1\}$, updated iteratively according to its local field:

$$
s_i \;\leftarrow\; \operatorname{sign}\!\Big(\sum_{j \neq i} J_{ji}\, s_j + J_D s_i \Big).
$$

Extending to a multilayer chain, each layer $l$ with state $s^{(l)}$ also receives excitatory inputs from its neighbors with coupling $\lambda$:

$$
s^{(l)}_i \;\leftarrow\; \operatorname{sign}\!\Big(\sum_{j \neq i} J^{(l)}_{ji}\, s^{(l)}_j \;+\; J_D s^{(l)}_i \;+\; \lambda (s^{(l-1)}_i + s^{(l+1)}_i) \Big).
$$

Our modules provide a software abstraction of this process.

* **Layers** compute the recurrent/self contributions ($J$, $J_D$) and handle aggregation + activation.
* **Adapters** contribute the cross-layer terms (e.g., $\lambda s^{(l\pm 1)}$).

In other words, for a layer $l$ with current state $s^{(l)}$, a typical pre-activation is

$$
h_i \;=\; \underbrace{\sum_{j \neq i} J_{ji}\, s_j + J_D s_i}_{\text{layer self-message}}
\;+\; \underbrace{\lambda\, s_i^{(l-1)} + \lambda\, s_i^{(l+1)}}_{\text{adapter messages}}
$$

and the new state is $s^{(l)} \leftarrow \text{activation}(h)$.

* The **self part** is computed by the layer’s `__call__`.
* The **cross-layer parts** are produced by adapters’ `__call__` on neighboring (or otherwise connected) states.
* The layer’s `reduce` performs the final aggregation into $h$.


---

## 2. Interfaces

All modules extend a common base:

```python
class AbstractModule(eqx.Module, ABC):

    @property
    @abstractmethod
    def has_state(self) -> bool: ...

    @abstractmethod
    def __call__(self, x: Array, rng: Array | None = None) -> Array: ...

    @abstractmethod
    def backward(self, x: Array, y: Array, y_hat: Array) -> Self: ...
```

* `__call__`: forward computation (a message).
* `backward`: returns a **module-shaped update**. These are not gradients; they are local plasticity rules.

### Layers

```python
class Layer(AbstractModule, ABC):

    @property
    def has_state(self) -> bool: return True

    @abstractmethod
    def activation(self, x: Array) -> Array: ...

    @abstractmethod
    def reduce(self, h: PyTree) -> Array: ...
```

* `activation`: nonlinearity applied to the aggregated field.
* `reduce`: combines incoming messages into a single tensor.

### Adapters

```python
class Adapter(AbstractModule, ABC):

    @property
    def has_state(self) -> bool: return False
```

Adapters transform a source state into a message for another layer.

---

## 3. Why layer states are “messages”

In this architecture, a layer’s **current state** is not just a transient activation, but the **signal it emits to the rest of the network**. Every update step consists of:

1. Each layer publishing its state as a message.
2. Adapters converting these messages into forms suitable for their targets.
3. Layers aggregating self-messages and incoming adapter messages into $h$.
4. Layers applying their activation to obtain the new state.

Thus, the global state itself is the medium of communication: states *are* messages.

---

## 4. Example modules

### RecurrentDiscrete

```python
class RecurrentDiscrete(Layer):
    J: Array
    J_D: Array
    threshold: Array

    def activation(self, x: Array) -> Array:
        return jnp.where(x >= 0, 1, -1).astype(x.dtype)

    def __call__(self, x: Array, rng=None) -> Array:
        return x @ self.J

    def reduce(self, h: PyTree) -> Array:
        return jnp.asarray(tree_reduce(operator.add, h))

    def backward(self, x: Array, y: Array, y_hat: Array) -> Self:
        dJ = perceptron_rule_backward(x, y, y_hat, self.threshold)
        zero_update = jax.tree.map(jnp.zeros_like, self)
        return eqx.tree_at(lambda m: m.J, zero_update, dJ)
```

* `__call__`: computes the recurrent/self-message.
* `reduce`: aggregates all incoming messages.
* `activation`: enforces ±1 states.
* `backward`: computes a local perceptron-like rule to update $J$.

### Ferromagnetic adapter

```python
class Ferromagnetic(Adapter):
    strength: Array

    def __call__(self, x: Array, rng=None) -> Array:
        return x * self.strength

    def backward(self, x, y, y_hat) -> Self:
        return tree_map(jnp.zeros_like, self)
```

A fixed adapter implementing terms like $\lambda s^{(l-1)}$. Since it has no trainable parameters, the `backward` methods returns an array of zeros,
meaning that the strength lambda remains unchanged.

---

## 5. One update step

```python
# state is a sequential state as shown in the first tutorial.
# layer is our recurrent (or any other) implementation of the Layer
# left and right are two adapters

def one_step(state, layer, left, right, l=1):
    # we select the state of `layer`
    s_l = state[l]
    # we compute the self/recurrent update (message)
    msg_self = layer(s_l)
    # we compute the message from the left (s[l-1]) through the adapter
    msg_l = left(state[l-1])
    # we compute the message from the right (s[l+1]) through the adapter
    msg_r = right(state[l+1])
    # we aggregate the computed messages directed to the layer
    h = layer.reduce([msg_self, msg_l, msg_r])
    # we apply the non linearity to the result
    s_next_l = layer.activation(h)
    # we update the state at position l with the new value
    return state.replace_val(l, s_next_l), h
```

---

## 6. Training with Optax

Unlike gradient-based deep learning, learning here uses **local updates**. Each module’s `backward` returns a module-shaped update, which we feed to Optax as if it were a gradient. This lets us retain momentum, schedules, clipping, etc., while remaining gradient-free.

Be sure to check [Optax documentation](https://optax.readthedocs.io/en/latest/).

```python
import optax

# we define an optimizer such as adam
optim = optax.adam(1e-2)

# we use adam across our layer and adapters
params = (layer, left, right)
opt_state = optim.init(eqx.filter(params, eqx.is_inexact_array))

def train_step(params, opt_state, state):
    layer, left, right = params

    # we compute the new state and preactivation as above
    s1, h1 = one_step(state, layer, left, right)

    # the backward functions compute updates of each layer
    upd_layer = layer.backward(s1[1], s1[1], h1)
    upd_left  = left.backward (s1[0], s1[0], h1)
    upd_right = right.backward(s1[2], s1[2], h1)

    # we use equinox to insert updates in the correct shapes
    # this is a typical equinox/optax pattern
    pseudo_grads = (upd_layer, upd_left, upd_right)
    grads_f = eqx.filter(pseudo_grads, eqx.is_inexact_array)
    params_f = eqx.filter(params, eqx.is_inexact_array)

    # optax computes the updated parameters
    updates, opt_state = optim.update(grads_f, opt_state, params=params_f)

    # we apply them to our modules
    params = eqx.apply_updates(params, updates)
    return params, opt_state, s1
```

A training loop repeatedly calls `train_step`. Non-trainable adapters contribute zero updates, so they remain fixed.
As said, we won't have to deal with the _wiring_ of each layer directly. This will be handled by the orchestrator.
Specifically, we will just need to call backward from the orchestrator to have a single update pytree of all modules
in our network. Jax is totally transparent to the structure of our network, as long as it is a pytree.

We provide a useful abstraction for this, see `LayerMap`.

---

## 7. Multi-dimensional states

States can have arbitrary shapes: vectors `(B, F)` or image-like `(B, H, W, C)`. Layers and adapters generalize naturally as long as their operations are defined on those shapes. The **global-state** design also supports shared buffers, where multiple layers operate on slices of a single array.

---

## 8. Summary

* **Layers** are stateful modules: self-message → reduce → activation.
* **Adapters** are stateless, transforming states into messages.
* **States are messages**: each layer’s current state is the signal it emits.
* **Backward rules**: modules return structured updates, not gradients.
* **Optax integration**: these updates are fed as pseudo-gradients to Optax, enabling optimizer dynamics without autodiff.
* **Global state**: supports vector and multi-dimensional architectures with shared memory.

This architecture mirrors the philosophy of distributed, local, gradient-free learning in asymmetric recurrent networks.
