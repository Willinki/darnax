import logging
from typing import Iterable, List, Optional
import equinox as eqx
import jax.numpy as jnp
from jax.typing import Array
from src.states.interface import State

logger = logging.getLogger(__name__)


class SequentialState(State):
    """
    a simple sequential state for layered networks
    items are accessed by layer index.

    - states: a list of arrays containing the state at
    layer l, where 0 is the input and L+1 is the output
    - dtype: default dtype of the state
    """

    states: List[Array]
    dtype: eqx.field(static=True)

    def __init__(self, sizes: Iterable[int], dtype=jnp.float32):
        """
        initializes a simple sequential state for layered networks
        items are accessed by layer index.

        args
        ----
        - sizes: Iterable[int]. An iterable of layer sizes, this includes
        both input and output.
        For example a one layer network will be initialized with [D, N, C].
        -dtype: jnp.dtype. The default type of the state.
        """
        assert all(
            isinstance(size, int) and size > 0 for size in sizes
        ), f"Not all provided sizes are positive ints. {sizes}"
        logger.info(f"Initializing SequentialState with layer sizes {sizes}")
        self.dtype = dtype
        self.states = [jnp.zeros(shape=(1, size), dtype=dtype) for size in sizes]

    def __getitem__(self, key: int) -> Array:
        return self.states[key]

    def init(self, x: Array, y: Optional[Array] = None) -> "SequentialState":
        """sets layer 0 to the input and layer -1 to the output, the rest is zero"""
        # --- shape checks at trace-time (run once per compilation) ---
        assert x.ndim == 2, f"x must be (B, D); got {x.shape}"
        sizes = tuple(arr.shape[1] for arr in self.states)
        assert (
            x.shape[1] == sizes[0]
        ), f"x.shape[1] != input size: {x.shape[1]} != {sizes[0]}"
        if y is not None:
            assert y.ndim == 2, f"y must be (B, C); got {y.shape}"
            assert (
                y.shape[0] == x.shape[0]
            ), f"batch mismatch: {x.shape[0]} vs {y.shape[0]}"
            assert (
                y.shape[1] == sizes[-1]
            ), f"y.shape[1] != output size: {y.shape[1]} != {sizes[-1]}"

        # --- build new (B, size_l) buffers for all layers ---
        b = x.shape[0]
        new_states = [jnp.zeros((b, s), dtype=self.dtype) for s in sizes]
        new_states[0] = x
        if y is not None:
            new_states[-1] = y
        # Replace the entire list functionally (no in-place mutation)
        return eqx.tree_at(lambda m: m.states, self, new_states)
