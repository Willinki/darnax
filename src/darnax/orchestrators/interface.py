"""Abstract orchestrator interface.

An orchestrator schedules **message passing**, **state updates**, and **local
learning** over a :class:`~darnax.layer_maps.sparse.LayerMap`. It owns no
parameters itself; it *routes* tensors through modules, updates a global
:class:`~darnax.states.interface.State`, and returns parameter updates shaped
like the layermap.

Responsibilities
----------------
- **step**: run one forward/update pass that refreshes non-output buffers
  (keeps the output buffer unchanged). Typically used inside a recurrent loop.
- **step_inference**: like ``step`` but restricted to a one-sided schedule
  (e.g., skip “right-going” messages) for cheaper partial updates; output
  remains unchanged.
- **predict**: compute the output buffer from the current internal buffers
  (no learning).
- **backward**: compute a PyTree of parameter updates with the **same structure**
  as the layermap (suitable for Optax ``update/apply_updates``).

RNG handling
------------
Each method accepts an RNG key and returns a **new** RNG key. Implementations
should **split** the key to ensure reproducible, side-effect-free randomness.

See Also
--------
tutorials/05_orchestrators.md
tutorials/06_simple_net_on_artificial_data.md

"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, Self, TypeVar

import equinox as eqx
import jax

from darnax.layer_maps.sparse import LayerMap

if TYPE_CHECKING:
    from darnax.modules.interfaces import AbstractModule
    from darnax.states.interface import State

    ModuleT = TypeVar("ModuleT", bound="AbstractModule")

StateT = TypeVar("StateT", bound="State")
KeyArray = jax.Array


class AbstractOrchestrator(eqx.Module, Generic[StateT]):
    """Handle communication and scheduling over a :class:`LayerMap`.

    The orchestrator defines *how* modules exchange messages (graph traversal,
    fan-in reduction, activation timing) and *when* learning rules are applied.
    Concrete subclasses encode the schedule (e.g., left→right passes, recurrent
    sweeps, or sparsity-aware traversals).

    Attributes
    ----------
    lmap : LayerMap
        The static adjacency (rows/cols and order). Module **parameters** inside
        the layermap are PyTree leaves visible to JAX/Optax; the key layout is
        static for JIT stability.

    Notes
    -----
    - Methods are expected to be **pure** with respect to inputs: given a state
      and RNG, they return a new state and a new RNG (no in-place mutation).
    - Use :func:`jax.random.split` to manage randomness. Determinism under JIT
      requires not reusing RNG keys.
    - ``backward`` returns *updates* (same PyTree shape as ``lmap``), not new
      parameters. Applying those updates is the optimizer’s job.

    """

    lmap: LayerMap

    @abstractmethod
    def step(self, state: StateT, *, rng: jax.Array) -> tuple[StateT, KeyArray]:
        """Run one forward/update step **without** touching the output buffer.

        Typical use is inside a recurrent loop to evolve hidden buffers while
        deferring the explicit output computation to :meth:`predict`.

        Parameters
        ----------
        state : StateT
            Current global state (input at index ``0``, output at ``-1``).
        rng : jax.Array
            PRNG key for any stochastic modules. Implementations should split
            the key internally.

        Returns
        -------
        (new_state, new_rng) : tuple[StateT, jax.Array]
            The updated state (same type) and a fresh RNG key.

        Notes
        -----
        - Must **not** modify ``state[-1]`` (the output buffer).
        - Scheduling is defined by the subclass (e.g., process edges with
          ``j <= i`` only, or a full sweep excluding output).

        """
        pass

    @abstractmethod
    def step_inference(self, state: StateT, *, rng: jax.Array) -> tuple[StateT, KeyArray]:
        """Run a cheaper, inference-oriented step (output unchanged).

        This variant avoids computing messages that travel “to the right”
        (exact meaning is schedule-specific; commonly skip edges with
        ``j > i``). Useful for partial refreshes of upstream buffers.

        Parameters
        ----------
        state : StateT
            Current global state.
        rng : jax.Array
            PRNG key to split.

        Returns
        -------
        (new_state, new_rng) : tuple[StateT, jax.Array]
            The updated state and a fresh RNG key.

        Notes
        -----
        Must **not** modify the output buffer ``state[-1]``.

        """
        pass

    @abstractmethod
    def predict(self, state: StateT, *, rng: jax.Array) -> tuple[StateT, KeyArray]:
        """Compute/refresh the **output** buffer from current internal buffers.

        Parameters
        ----------
        state : StateT
            Current global state (uses existing hidden buffers).
        rng : jax.Array
            PRNG key to split, if output computation is stochastic.

        Returns
        -------
        (new_state, new_rng) : tuple[StateT, jax.Array]
            A state where ``state[-1]`` has been updated, and a fresh RNG key.

        Notes
        -----
        This method performs **no parameter updates**; it’s a pure readout pass.

        """
        pass

    @abstractmethod
    def backward(self, state: StateT, rng: KeyArray) -> Self:
        """Compute parameter updates aligned with the layermap structure.

        Parameters
        ----------
        state : StateT
            Current global state providing the activations/messages required by
            local learning rules (e.g., perceptron updates).
        rng : jax.Array
            PRNG key to split for stochastic update rules.

        Returns
        -------
        Self
            A PyTree **with the same structure as** ``self`` where the ``lmap``
            subtree mirrors the original layermap but each module is replaced by
            its parameter **update** (zeros for non-trainable fields).

        Notes
        -----
        - The returned object is intended for use with Optax:
          ``updates = orchestrator.backward(state, rng)``,
          then ``new_params = optax.apply_updates(params, updates)``.
        - Implementations should avoid side effects and return only *updates*,
          not updated parameters.

        """
        pass
