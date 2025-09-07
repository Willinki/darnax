import operator
from functools import reduce
import jax
from jax.typing import Array, PyTree
from src.modules.interfaces import Layer


class OutputLayer(Layer):
    """Mostly placeholder adapter, might do
    complex things in the future"""

    @jax.jit
    def __call__(self, x: Array) -> Array:
        return x

    @jax.jit
    def reduce(self, h: PyTree) -> Array:
        leaves = jax.tree.leaves(h)
        N = len(leaves)
        return jax.nn.softmax(reduce(operator.add, leaves) / N)
