from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jax import Array, lax

from darnax.utils.typing import PyTree


def scan_n(
    f: Callable[..., Any],
    init: Any,
    n_iter: int,
    *f_args: Any,
    **f_kwargs: Any,
) -> tuple[Any, Any]:
    r"""Run a step function for ``n_iter`` iterations using ``jax.lax.scan``.

    The step function is called as ``f(carry, *f_args, **f_kwargs)`` each
    iteration. Its per-step outputs (the second return value) are stacked
    along a new leading time dimension by ``lax.scan`` and returned.

    Parameters
    ----------
    f : callable
        Step function with signature
        ``f(carry, *f_args, **f_kwargs) -> (new_carry, out)``.
        ``carry`` and ``out`` can be any pytrees of JAX arrays
        (e.g., tuples like ``(rng, state)``, dictionaries, dataclasses).
    init : pytree
        Initial carry (pytree of arrays).
    n_iter : int
        Number of iterations to run.
    *f_args : Any
        Positional arguments forwarded to ``f`` on every step.
    **f_kwargs : Any
        Keyword arguments forwarded to ``f`` on every step.

    Returns
    -------
    carry : pytree
        Final carry after ``n_iter`` steps (e.g., final ``(rng, state)``).
    outputs : pytree or None
        Time-stacked outputs from each step (leading dimension = ``n_iter``).
        If ``f`` returns ``None`` as the output, this will be ``None``.

    Notes
    -----
    - Implemented as ``lax.scan(body, init, xs=None, length=n_iter)``.
    - Non-array values in ``f_args``/``f_kwargs`` are treated as **static**;
      changing their identity will retrace/JIT-recompile. Values that vary
      should be JAX arrays.
    - ``carry`` may be any pytree, including tuples like ``(rng, state)``.

    Examples
    --------
    Carry as ``(rng, state)`` where both are pytrees:

    >>> import jax.numpy as jnp
    >>> from jax import random
    >>> # toy "state" as a dict; in your code this is your State object/pytree
    >>> def step(carry, lr):
    ...     rng, state = carry
    ...     rng, sub = random.split(rng)
    ...     # pretend "state['w']" is a parameter; do a dummy update
    ...     grad | None = None         # fake gradient
    ...     new_w = state["w"] - lr * grad
    ...     new_state = {"w": new_w}
    ...     loss = new_w ** 2             # fake loss to log
    ...     return (rng, new_state), loss
    ...
    >>> init_rng = random.PRNGKey(0)
    >>> init_state = {"w": jnp.array(3.0)}
    >>> (final_rng, final_state), losses = scan_n(step, (init_rng, init_state), n_iter=5, lr=jnp.array(0.1))
    >>> final_state["w"]  # moved 5 steps of size 0.1 from 3.0
    Array(2.5, dtype=float32)
    >>> losses.shape  # collected per-step outputs
    (5,)

    """

    def body(carry: PyTree, _: Any) -> tuple[PyTree, Any]:
        new_carry = f(*carry, *f_args, **f_kwargs)
        return new_carry, _

    final_carry, _ = lax.scan(body, init, xs=None, length=n_iter)
    return final_carry, _


def batch_accuracy(y_true: Array, y_pred: Array) -> Array:
    """Compute accuracy with ±1 OVA labels (class = argmax along last dim)."""
    y_true_idx = jnp.argmax(y_true, axis=-1)
    y_pred_idx = jnp.argmax(y_pred, axis=-1)
    return jnp.mean((y_true_idx == y_pred_idx).astype(jnp.float32))
