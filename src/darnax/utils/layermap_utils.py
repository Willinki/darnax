from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax

from darnax.layer_maps.sparse import LayerMap

if TYPE_CHECKING:
    from collections.abc import Callable


def layermap_apply(
    f: Callable[[jax.Array], Any],
    select_idxs: Callable[[tuple[int, int]], bool],
    lmap: LayerMap,
) -> LayerMap:
    """Apply a transformation to *parameter arrays only* in selected modules of a LayerMap.

    This variant first partitions the LayerMap into its trainable (array) and static
    components using :func:`equinox.partition`, applies the function ``f`` exclusively
    to arrays inside modules selected by ``select_idxs``, and finally re-combines the
    two parts with :func:`equinox.combine`. Non-array leaves (e.g. constants, shapes,
    callables, buffers, RNG keys) remain untouched and never enter the JAX tracing path.

    Parameters
    ----------
    f : Callable[[jax.Array], Any]
        Function applied to every *array* leaf of selected modules. Must be
        compatible with JAX transformations (e.g., pure and shape-preserving)
        if you intend to ``jit`` the caller.

    select_idxs : Callable[[tuple[int, int]], bool]
        Predicate determining which modules to transform.
        Receives a tuple ``(i, j)`` corresponding to the receiver and sender
        indices in the LayerMap entry ``lmap[i][j]``.
        It should return ``True`` for modules that should be transformed.
        Example::
            lambda ij: ij[0] == ij[1]   # select only diagonal modules

    lmap : LayerMap
        The LayerMap (a dict-of-dicts pytree) whose values are Equinox modules.
        Its structure is typically static, so the transformation preserves the same
        nested layout.

    Returns
    -------
    LayerMap
        A new LayerMap where ``f`` has been applied to all parameter arrays of each
        selected module, while all non-selected modules and non-array leaves remain
        unchanged. The original object is never mutated.

    Notes
    -----
    • The function is *purely functional* — it constructs and returns a new pytree.
      The input ``lmap`` is left intact.

    • Partitioning with :func:`equinox.partition` guarantees that only numerical
      leaves are traversed by :func:`jax.tree_map`. This prevents Python objects or
      static data from entering JIT-compiled traces.

    • This pattern mirrors common Equinox practice for separating model parameters
      from static state before optimization or device transfer.

    Examples
    --------
    >>> # Scale all parameters of diagonal modules by 0.5
    >>> new_lmap = layermap_apply_params_only(
    ...     f=lambda x: 0.5 * x,
    ...     select_idxs=lambda ij: ij[0] == ij[1],
    ...     lmap=lmap,
    ... )

    >>> # Move parameters of upper-triangular modules to float32
    >>> new_lmap = layermap_apply_params_only(
    ...     f=lambda x: x.astype(jnp.float32),
    ...     select_idxs=lambda ij: ij[0] < ij[1],
    ...     lmap=lmap,
    ... )

    """
    # 1) Separate numerical parameters from non-array statics.
    params, statics = eqx.partition(lmap, eqx.is_array)

    # 2) Build a same-structure boolean mask marking selected modules.
    mask = {i: {j: bool(select_idxs((i, j))) for j in row} for i, row in lmap.row_items()}

    # 3) Transform parameter leaves only (statics untouched).
    def _apply(x: Any, m: bool) -> Any:
        # x: an array leaf from params
        # m: bool at module level, broadcast to its leaves
        return f(x) if m else x

    new_params = jax.tree_map(_apply, params, mask)

    # 4) Recombine trainable and static components into a new LayerMap.
    combined_params = eqx.combine(new_params, statics)
    return LayerMap.from_dict(combined_params, require_diagonal=True)
