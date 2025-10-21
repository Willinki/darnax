# darnax


[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://dbadalotti.com/darnax)
[![arXiv](https://img.shields.io/badge/arXiv-2509.05041-b31b1b.svg)](https://arxiv.org/abs/2509.05041)

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
👉 [dbadalotti.com/darnax](https://dbadalotti.com/darnax)

---

## Contributing

This project is a work in progress — contributions, issues, and discussions are welcome!

## Citing

If you use **darnax** in your research, please cite the following work:

> Davide Badalotti, Carlo Baldassi, Marc Mézard, Mattia Scardecchia, Riccardo Zecchina.
> *Dynamical Learning in Deep Asymmetric Recurrent Neural Networks*.
> arXiv:2509.05041 (2025).

```bibtex
@article{badalotti2025darnax,
  title={Dynamical Learning in Deep Asymmetric Recurrent Neural Networks},
  author={Badalotti, Davide and Baldassi, Carlo and Mézard, Marc and Scardecchia, Mattia and Zecchina, Riccardo},
  journal={arXiv preprint arXiv:2509.05041},
  year={2025}
}
```


