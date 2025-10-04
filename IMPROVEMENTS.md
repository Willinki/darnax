# Improvements

This file is made to list future improvements of the code base.

It is supposed to contain high level ideas, for specific fixes, open an issue (well actually...)

- [ ] Remove sequential key splitting in the orchestrator

- [ ] Define a batched state, where instead of lists we have a tensor to support vmap(switch(...)) in the orchestrator launch.
It should handle padding and masking. I guess it should be a separate interface.

- [x] Introduce support for equinox.partition in Layermap, right now it does not work. In general we should test all object against
equinox partition/combine.

- [ ] Add validation function to validate shapes before training.

- [ ] Consider the addition of a placeholder "input_layer" to retain consistency.

- [ ] Add helper function to build networks

- [ ] How to handle parameter sharing? eqx.shared

- [ ] Orchestrator should be generic type in both state and layermap

- [ ] Add Trainer object to remove further boilerplate

- [ ] Improve docs in perceptron rule update docs to clear meaning and clarify J x against J s in the paper

- [ ] Add system to mask parameters for optax
