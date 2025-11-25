"""Sequential, array-backed global state.

`SequentialState` stores per-layer activations for layered networks as a list of
JAX arrays and follows a strict indexing convention:

- index ``0``: input buffer
- index ``-1``: output buffer
- indices ``1..-2``: intermediate layer buffers (left → right)

Construction allocates **placeholder** buffers with batch size 1. Call
:meth:`SequentialState.init` to resize all buffers to the true batch size and
populate the endpoints. The object is an Equinox ``Module`` (a PyTree), so it
plays nicely with JAX transforms; all “updates” are **functional** (return a new
instance) via :meth:`replace` and :meth:`replace_val`.
"""

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

    Layers are indexed from left to right: ``state[0]`` is **input** and
    ``state[-1]`` is **output**. The internal storage is a list of arrays with
    shapes ``(B, *size_l)`` where ``B`` is the batch size and ``*size_l`` is the
    per-layer trailing shape.

    On construction, buffers are initialized as zeros with shape
    ``(1, *size_l)`` (a placeholder batch). Call :meth:`init` to resize all
    buffers to a real batch size and optionally set the output.

    Attributes
    ----------
    states : list[Array]
        Per-layer buffers. Shapes are ``(B, *size_l)`` after :meth:`init`,
        or ``(1, *size_l)`` immediately after construction.
    dtype : jnp.dtype
        Dtype for all buffers (marked static in the PyTree).
    data_min_ndim : int
        Minimal rank for data buffers (default ``2`` → enforces a batch dim).

    Notes
    -----
    - This class is **functional**: methods like :meth:`replace` and
      :meth:`replace_val` return a **new** instance; the original is unchanged.
    - Shape assertions in :meth:`init` are executed at trace time, so they are
      checked once per JIT compilation.
    - Nothing prevents you from storing non-floating dtypes, as long as they
      are consistent with ``dtype`` at construction.

    Examples
    --------
    >>> st = SequentialState([4, 8, 2])        # input(4), hidden(8), output(2)
    >>> st = st.init(jnp.zeros((32, 4)))       # B=32; output left as zeros
    >>> x = st[0]                               # (32, 4)
    >>> st2 = st.replace_val(1, jnp.ones((32, 8)))  # update hidden layer

    """

    states: list[Array]
    messages: list[Array]
    dtype: jnp.dtype = eqx.field(static=True)
    data_min_ndim: int = eqx.field(default=2, static=True)

    def __init__(
        self, sizes: Iterable[tuple[int, ...] | int], dtype: jnp.dtype = jnp.float32
    ) -> None:
        """Create a sequential state with one buffer per layer.

        Parameters
        ----------
        sizes : Iterable[tuple[int, ...] | int]
            Iterable of **positive** sizes for each layer, including input and
            output. Each entry can be a positive ``int`` (for 1D layers) or a
            tuple of positive ``int`` (for multi-axis layers). Examples:
            ``[D, N, C]`` or ``[(H, W, C_in), N, (C_out,)]``.
        dtype : jnp.dtype, optional
            Dtype for all buffers. Default is ``jnp.float32``.

        Raises
        ------
        AssertionError
            If ``sizes`` is empty, or if any entry is not a positive integer or
            a tuple of positive integers.

        """
        sizes_list = list(sizes)
        assert len(sizes_list) > 0, "sizes must be a non-empty iterable."

        shape_tuples = [self._to_shape_tuple(s) for s in sizes_list]

        logger.info("Initializing SequentialState with layer sizes %s", shape_tuples)
        self.dtype = dtype
        # Start with (1, *size) buffers. Call `init` to match a real batch size.
        self.states = [jnp.zeros((1, *size), dtype=dtype) for size in shape_tuples]
        self.messages = [jnp.zeros((1, *size), dtype=dtype) for size in shape_tuples]

    def __len__(self) -> int:
        """Return the number of layers (including input and output)."""
        return len(self.states)

    def __getitem__(self, key: int) -> Array:
        """Return the buffer for layer ``key`` (supports negative indices).

        Parameters
        ----------
        key : int
            Layer index. ``0`` is input, ``-1`` is output. Negative indices
            follow Python semantics.

        Returns
        -------
        Array
            The requested buffer with shape ``(B, *size_key)`` (or ``(1, *size)``
            before :meth:`init`).

        """
        return self.states[key]

    def replace(self, value: Sequence[Array] | PyTree) -> Self:
        """Return a new instance with ``states`` replaced by ``value``.

        Parameters
        ----------
        value : Sequence[Array] | PyTree
            A sequence (or PyTree) that will replace the internal list of
            buffers. In typical use, a list of arrays of length ``len(self)``,
            each shaped ``(B, *size_l)``.

        Returns
        -------
        Self
            A new ``SequentialState`` carrying ``value`` as its storage.

        Notes
        -----
        This is a **functional** update; the original object is not mutated.

        """
        new_self: Self = eqx.tree_at(lambda s: s.states, self, value)
        return new_self

    def replace_message_val(self, idx: int, value: Array) -> Self:
        """Return a new instance with message buffer ``idx`` replaced by ``value``.

        This mirrors :meth:`replace_val` but targets the per-layer message/momentum
        buffers stored in ``self.messages``.
        """
        new_self: Self = eqx.tree_at(lambda s: s.messages[idx], self, value)
        return new_self

    def replace_val(self, idx: int, value: Array) -> Self:
        """Return a new instance with layer ``idx`` replaced by ``value``.

        Parameters
        ----------
        idx : int
            Layer index to modify.
        value : Array
            New buffer for that layer, typically with shape ``(B, *size_idx)``.

        Returns
        -------
        Self
            A new instance where only the selected layer differs from ``self``.

        """
        new_self: Self = eqx.tree_at(lambda s: s.states[idx], self, value)
        return new_self

    def init(self, x: Array, y: Array | None = None) -> Self:
        """Resize buffers to batch ``B`` and set input (and optionally output).

        Parameters
        ----------
        x : Array
            Input batch with shape ``(B, *D)``; the trailing shape ``*D`` must
            equal the configured input layer shape.
        y : Array or None, optional
            Optional output batch with shape ``(B, *C)``; the trailing shape
            ``*C`` must equal the configured output layer shape. If omitted,
            the output buffer is zero-initialized.

        Returns
        -------
        Self
            A new instance whose buffers all have shape ``(B, *size_l)``,
            with layer ``0`` set to ``x`` and (if provided) layer ``-1`` set to
            ``y``.

        Raises
        ------
        AssertionError
            If shapes are inconsistent (checked at trace time under JIT).

        Notes
        -----
        This method does **not** mutate the receiver; it constructs new buffers
        with batch size ``B`` and returns a fresh state.

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
        new_messages = [jnp.zeros((b, *shape), dtype=self.dtype) for shape in layer_shapes]
        new_self: Self = eqx.tree_at(lambda m: m.states, self, new_states)
        new_self = eqx.tree_at(lambda m: m.messages, new_self, new_messages)
        return new_self

    @property
    def readout(self) -> Array:
        """Return the readout state."""
        return self[-1]

    @property
    def representations(self) -> Array:
        """Return the representations (final hidden layer) state."""
        return self[-2]

    @staticmethod
    def _to_shape_tuple(s: tuple[int, ...] | int) -> tuple[int, ...]:
        """Validate and convert a size spec to a shape tuple.

        Parameters
        ----------
        s : tuple[int, ...] or int
            A positive integer for 1D layers, or a tuple of positive integers
            for multi-axis layers.

        Returns
        -------
        tuple[int, ...]
            The validated layer shape as a tuple.

        Raises
        ------
        AssertionError
            If ``s`` is neither a positive integer nor a tuple of positive
            integers.

        """
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
