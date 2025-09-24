# darnax

**Deep Asymmetric Recurrent Networks in JAX.**

darnax is a research library for building and experimenting with **asymmetric recurrent neural networks** and their learning dynamics. It is inspired by [recent work](https://arxiv.org/html/2509.05041v1) by Badalotti, Baldassi, Mézard, Scardecchia and Zecchina, showing that simple, distributed plasticity rules can give rise to powerful learning behaviors without relying on backpropagation.

The library provides:

* **Composable modules** based on [Equinox](https://github.com/patrick-kidger/equinox), with clean support for sparse and structured connectivity. They can be easily extended to accomodate for complex network structures (convolutional, transformers-like, etc...). They also naturally support layers of varying shapes and connectivity.
* **Orchestrators** for running recurrent dynamics, either sequentially or in parallel.
* **Local update rules** implementing gradient-free plasticity mechanisms.
* **Jax speed and transparency**. Everything is a pytree, whether you are building a simple 1 hidden-layer model or a complex interconnected structure, the training logic remains the same.
* **Natural integration with optax**. Despite not relying on explicit gradients, the models can be naturally optimized with [Optax](https://optax.readthedocs.io/en/latest/).

darnax is not a framework chasing SOTA benchmarks. It is a **sandbox** for exploring recurrent dynamics as a computational primitive — bridging machine learning, theoretical neuroscience, and statistical physics. This is also a work-in-progress, and contributions are more than welcome!

---

👉 Check the [Tutorials](./tutorials) to get started, or browse the [API Reference](./reference) for details.

---
