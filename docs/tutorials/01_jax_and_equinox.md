# 01 — JAX, Equinox, and Pytrees

This library is based on two foundational tools: **[JAX](https://jax.readthedocs.io/)** and **[Equinox](https://docs.kidger.site/equinox/)**.
Understanding their philosophy is essential for working with this codebase.
Both libraries share a unifying principle: **everything is a pytree**.

This tutorial provides a concise introduction to these ideas. Readers are encouraged to consult the official documentation of JAX and Equinox for further technical details.

---

## JAX: Transformations of Numerical Functions

[JAX](https://jax.readthedocs.io/) extends NumPy with support for automatic differentiation and compilation. Its design emphasizes **functional programming** and **composable transformations**.

Key concepts include:

* **Array programming**: JAX arrays follow NumPy semantics while executing efficiently on CPU, GPU, or TPU.
* **Transformations**: JAX operates by transforming functions:

  * `jax.jit` compiles Python functions to optimized machine code.
  * `jax.grad` computes derivatives automatically.
  * `jax.vmap` vectorizes functions across batch dimensions.
  * `jax.pmap` parallelizes computations across multiple devices.
* **Composability**: Transformations may be freely combined. For example, one may compute gradients of a JIT-compiled function or vectorize a function that already involves differentiation.

The emphasis is not on predefined models or layers, but on *transformations of user-defined functions*.

JAX gives the user a lot of powers, but this comes at a cost. If you're new to jax, be sure to read the [Sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

---

## Equinox: Pytrees as Models

[Equinox](https://docs.kidger.site/equinox/) is a lightweight neural network library for JAX. Its primary contribution is a consistent interface for defining models as **plain Python classes**.

The design principles of Equinox are:

* **Simplicity**: Models are standard Python objects.
* **Transparency**: Parameters are stored directly as object attributes.
* **Compatibility**: Models are implemented as pytrees, allowing them to participate seamlessly in JAX transformations.
* **Functional style**: Updates to parameters or state return new objects, rather than mutating existing ones.

This perspective aligns with the philosophy of this library: abstractions remain minimal, while full compatibility with JAX is preserved.

---

## Pytrees: A Unifying Abstraction

A **pytree** is a nested structure composed of Python containers (lists, tuples, dictionaries, dataclasses, and similar types) whose leaves are JAX arrays or compatible objects.

Examples of pytrees:

```python
import jax.numpy as jnp

# Dictionary with arrays
x = {"a": jnp.ones((2, 2)), "b": jnp.zeros((3,))}

# Tuple of arrays
y = (jnp.arange(3), jnp.ones((2,)))

# Nested structures
z = [x, y]
```

Pytrees are central to JAX for two reasons:

1. They generalize beyond single arrays to arbitrary nested data structures.
2. They allow JAX transformations to operate uniformly over these structures.

In practice:

* A model is a pytree.
* Parameters and optimizer states are pytrees.
* Data batches may also be represented as pytrees.

This single abstraction ensures that JAX transformations (such as `grad` or `jit`) can be applied consistently, regardless of structural complexity.

---

## Example: A Linear Module

Equinox makes use of the pytree abstraction by treating models as pytrees.

```python
import jax
import jax.numpy as jnp
import equinox as eqx

class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_dim, out_dim, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (in_dim, out_dim))
        self.bias = jax.random.normal(bkey, (out_dim,))

    def __call__(self, x):
        return x @ self.weight + self.bias

# Instantiate a model
model = Linear(2, 3, jax.random.PRNGKey(0))

# Forward evaluation
x = jnp.ones((5, 2))
y = model(x)

# Differentiation
loss_fn = lambda m, x: jnp.mean(m(x))
grads = jax.grad(loss_fn)(model, x)
```

Here, the model is both:

* a Python object with fields (`weight`, `bias`), and
* a pytree, enabling JAX to compute gradients and apply transformations directly.

---

## Philosophy of This Library

The present library is guided by the following principles:

1. **Universality of pytrees**: all major components (modules, layermaps, states) are structured as pytrees.
2. **Functional style**: computations are expressed as pure functions, and updates return new objects.
3. **Composability**: any component should be compatible with JAX transformations such as `jit`, `grad`, or `vmap`.
4. **Minimal abstraction**: the library extends JAX and Equinox without concealing them. Users are encouraged to understand and directly employ these underlying tools.

---

## Next Tutorial

The next tutorial will discuss the first component of this library: states and modules.
