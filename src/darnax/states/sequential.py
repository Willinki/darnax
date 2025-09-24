from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from darnax.states.interface import State

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Self

    from jax import Array

    from darnax.utils.typing import PyTree

logger = logging.getLogger(__name__)


class SequentialState(State):
    """Sequential activations buffer for layered networks.

    Layers are indexed from left to right:
    layer ``0`` is the **input** buffer and layer ``-1`` is the **output** buffer.

    The internal storage is a list of JAX arrays with shapes ``(B, *size_l)``.
    At construction time buffers are initialized with ``(1, *size_l)`` zeros.
    Use :meth:`init` to resize them to a real batch size ``B`` and optionally set the output.
    """

    states: list[Array]
    dtype: jnp.dtype = eqx.field(static=True)
    data_min_ndim: int = eqx.field(default=2, static=True)

    def __init__(
        self, sizes: Iterable[tuple[int, ...] | int], dtype: jnp.dtype = jnp.float32
    ) -> None:
        """Create a sequential state with one buffer per layer.

        Parameters
        ----------
        sizes
            Iterable of **tuples of positive integers** or **positive integers**
            for each layer width, including input and output.
            For example, a one-hidden-layer classifier could be ``[D, N, C]``
            or mixed-rank like ``[(H, W, C_in), N, (C_out,)]``.
        dtype
            JAX dtype for all buffers (static in the PyTree).

        Raises
        ------
        AssertionError
            If any size is not a positive ``int``/tuple of positive ``int`` or
            if the iterable is empty.

        """
        sizes_list = list(sizes)
        assert len(sizes_list) > 0, "sizes must be a non-empty iterable."

        shape_tuples = [self._to_shape_tuple(s) for s in sizes_list]

        logger.info("Initializing SequentialState with layer sizes %s", shape_tuples)
        self.dtype = dtype
        # Start with (1, *size) buffers. Call `init` to match a real batch size.
        self.states = [jnp.zeros((1, *size), dtype=dtype) for size in shape_tuples]

    def __len__(self) -> int:
        """Get number of layers (including input and output)."""
        return len(self.states)

    def __getitem__(self, key: int) -> Array:
        """Return the buffer for layer ``key`` (supports negative indices)."""
        return self.states[key]

    def replace(self, value: Sequence[Array] | PyTree) -> Self:
        """Return a new instance with ``states`` replaced by ``value``.

        Notes
        -----
        This is a **functional** update; the original object is not mutated.

        """
        new_self: Self = eqx.tree_at(lambda s: s.states, self, value)
        return new_self

    def replace_val(self, idx: int, value: Array) -> Self:
        """Return a new instance with layer ``idx`` replaced by ``value``."""
        new_self: Self = eqx.tree_at(lambda s: s.states[idx], self, value)
        return new_self

    def init(self, x: Array, y: Array | None = None) -> Self:
        """Resize buffers to batch ``B`` and set input (and optionally output).

        Parameters
        ----------
        x
            Input batch with shape ``(B, *D)`` where ``*D`` must match the input layer shape.
        y
            Optional output batch with shape ``(B, *C)`` where ``*C`` must match the output layer shape.

        Returns
        -------
        SequentialState
            A new instance whose buffers all have shape ``(B, *size_l)``, with
            layer 0 set to ``x`` and (if provided) layer -1 set to ``y``.

        Raises
        ------
        AssertionError
            If shapes are inconsistent. Checks run at trace time (once per compilation).

        """
        # --- trace-time checks (once per JIT compilation) ---
        assert x.ndim >= self.data_min_ndim, f"x must be (B, *D); got {x.shape}"

        # shapes excluding batch dimension for each layer
        layer_shapes = tuple(arr.shape[1:] for arr in self.states)

        # input trailing dims must match input layer shape
        assert (
            x.shape[1:] == layer_shapes[0]
        ), f"x trailing dims {x.shape[1:]} != input layer shape {layer_shapes[0]}"

        if y is not None:
            assert y.ndim >= self.data_min_ndim, f"y must be (B, *C); got {y.shape}"
            assert y.shape[0] == x.shape[0], f"batch mismatch: {x.shape[0]} vs {y.shape[0]}"
            assert (
                y.shape[1:] == layer_shapes[-1]
            ), f"y trailing dims {y.shape[1:]} != output layer shape {layer_shapes[-1]}"

        # --- allocate new (B, *size_l) buffers and set endpoints ---
        b = x.shape[0]
        new_states = [jnp.zeros((b, *shape), dtype=self.dtype) for shape in layer_shapes]
        new_states[0] = x
        if y is not None:
            new_states[-1] = y
        new_self: Self = eqx.tree_at(lambda m: m.states, self, new_states)
        return new_self

    @staticmethod
    def _to_shape_tuple(s: tuple[int, ...] | int) -> tuple[int, ...]:
        """Perform checks and convert to suitable tuple."""
        if isinstance(s, int):
            assert s > 0, f"size must be positive int; got {s}"
            return (s,)
        elif isinstance(s, tuple):
            assert len(s) > 0 and all(
                isinstance(d, int) and d > 0 for d in s
            ), f"tuple sizes must be positive ints; got {s}"
            return s
        else:
            raise AssertionError(f"size entries must be int or tuple[int,...]; got {type(s)}")
