"""State interfaces.

This module defines the abstract :class:`State` contract used by orchestrators
and layer maps. A concrete state holds per-layer tensors plus two special
slots:

- index ``0``: **input** (provided by the caller),
- last index: **output** (produced by the network).

Implementations must be Equinox ``Module``s (i.e., PyTrees) and support
functional updates (returning **new** state instances rather than mutating
in place).

"""

from abc import abstractmethod, abstractproperty
from typing import Any, Self

import equinox as eqx
from jax import Array

from darnax.utils.typing import PyTree


class State(eqx.Module):
    """Global state for a network (LayerMap).

    A ``State`` stores the tensors associated with each layer plus the
    designated **input** and **output** slots. The total number of elements
    must be **at least** ``#layers + 1`` (input + one per layer + output).

    Concretely, orchestrators assume:
    - ``state[0]`` holds the **input**,
    - ``state[-1]`` holds the **output**,
    - the remaining indices map to intermediate layer states according to the
      LayerMap’s ordering.

    Notes
    -----
    - ``State`` is an Equinox ``Module`` and thus a PyTree; it should be safe
      to pass through JAX transformations (``jit``, ``vmap``) as long as
      implementations avoid in-place mutation.
    - All mutating operations must be exposed as **functional** methods that
      return a *new* instance (see :meth:`replace` and :meth:`replace_val`).

    See Also
    --------
    darnax.states.sequential.SequentialState
        A concrete reference implementation using array-backed storage.

    """

    @abstractmethod
    def __getitem__(self, key: Any) -> Array:
        """Return the array stored at ``key``.

        Parameters
        ----------
        key : Any
            Index/key understood by the concrete implementation (commonly an
            ``int`` or a tuple).

        Returns
        -------
        Array
            The stored tensor for the given position.

        Notes
        -----
        Implementations should raise ``IndexError``/``KeyError`` on invalid
        positions, mirroring standard Python container semantics.

        """

    @abstractmethod
    def init(self, x: Array, y: Array | None = None) -> Self:
        """Initialize input/output slots and return a fresh state.

        Parameters
        ----------
        x : Array
            Input tensor to place at index ``0``.
        y : Array or None, optional
            Optional target/output tensor to place at the last index. If
            ``None``, the output slot should be initialized according to the
            implementation’s policy (e.g., zeros or a placeholder).

        Returns
        -------
        Self
            A new state instance with input/output slots set.

        Notes
        -----
        This method must **not** mutate the receiver; it returns a new state
        object with the appropriate slots populated.

        """

    @abstractproperty
    def readout(self) -> Array:
        """Return the readout state."""

    @abstractmethod
    def replace(self, value: PyTree) -> Self:
        """Return a new state with **all** underlying values replaced.

        Parameters
        ----------
        value : PyTree
            A PyTree matching the internal storage structure.

        Returns
        -------
        Self
            A new instance containing ``value`` as its storage.

        Notes
        -----
        This is the coarse-grained replacement primitive used by orchestrators
        when recomputing the entire state in one shot.

        """

    @abstractmethod
    def replace_val(self, idx: Any, value: Array) -> Self:
        """Return a new state with a **single** slot replaced.

        Parameters
        ----------
        idx : Any
            Index/key indicating which slot to replace (commonly an ``int``).
        value : Array
            New tensor to store at the selected position.

        Returns
        -------
        Self
            A new instance where only the selected slot differs from ``self``.

        Notes
        -----
        This is the fine-grained primitive used during step-wise updates
        (e.g., updating a single layer’s activation while keeping the rest).

        """
