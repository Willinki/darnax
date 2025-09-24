# darnax

**Deep Asymmetric Recurrent Networks in JAX.**

darnax is a research library for building and experimenting with asymmetric recurrent neural networks and their learning dynamics. Inspired by [recent work](https://arxiv.org/html/2509.05041v1) on local plasticity and representational manifolds, 
it offers a lightweight, composable toolkit for studying distributed, gradient-free learning in deep recurrent models.

---

## Features

* **Composable modules** built on [Equinox](https://github.com/patrick-kidger/equinox), with support for sparse and structured connectivity.
* **Orchestrators** for sequential or parallel recurrent dynamics.
* **Local update rules** implementing gradient-free plasticity mechanisms.
* **Optax integration** for optimization, even without explicit gradients.
* **Pure JAX pytrees**: everything is transparent and functional.

---

## Installation

```bash
pip install git+https://github.com/Willinki/darnax.git
```

---

## Documentation

📖 Full documentation and tutorials are available at:
👉 [dbadalotti.com](https://dbadalotti.com)

---

## Contributing

This project is a work in progress — contributions, issues, and discussions are welcome!

---

Do you also want me to add a **"Citing" section** at the end, pointing to your arXiv paper, so researchers can reference the library?
