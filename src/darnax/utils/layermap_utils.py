from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx

from darnax.layer_maps.sparse import LayerMap

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax


from jax.tree_util import tree_flatten, tree_unflatten


def layermap_apply(
    f: Callable[[jax.Array], Any],
    select_idxs: Callable[[tuple[int, int]], bool],
    lmap: LayerMap,
) -> LayerMap:
    """Apply a transformation to *parameter arrays only* in selected modules of a LayerMap.

    This variant:
      1. Partitions the LayerMap into its trainable (array) and static parts using
         :func:`equinox.partition`.
      2. Iterates over each module `(i, j)` in the LayerMap.
      3. For selected modules (where ``select_idxs((i, j))`` is True):
         - Flattens their parameter pytree into a list of array leaves.
         - Applies ``f`` independently to each array leaf.
         - Reconstructs the module with :func:`jax.tree_util.tree_unflatten`.
      4. Reassembles the modified parameters with the untouched statics via
         :func:`equinox.combine`.

    This design is purely functional — no in-place edits, no side effects, and
    full compatibility with JAX transformations when the structure is static.

    Parameters
    ----------
    f : Callable[[jax.Array], Any]
        Function applied to each *array* leaf (parameter) of the selected modules.
        Must be elementwise or shape-preserving if you plan to `jit` the caller.
        The function can include arbitrary JAX operations.

    select_idxs : Callable[[tuple[int, int]], bool]
        Predicate that decides which modules are transformed.
        Receives the coordinate pair ``(i, j)`` representing the receiver and sender
        indices in the LayerMap entry ``lmap[i][j]``.
        It must return ``True`` for modules that should be processed.

        Example::
            lambda ij: ij[0] == ij[1]   # Select diagonal modules
            lambda ij: ij[0] < ij[1]    # Select upper-triangular modules

    lmap : LayerMap
        A LayerMap (dict-of-dicts pytree) whose values are Equinox modules.
        Each module can itself contain submodules and arrays.
        The structure is assumed to be *static* (not changing between calls).

    Returns
    -------
    LayerMap
        A new LayerMap in which the array leaves of selected modules have been
        transformed by ``f``. Non-selected modules and all non-array leaves
        (statics, metadata, etc.) remain unchanged.

    Notes
    -----
    • Flattening and unflattening is done *per module*, not globally. This avoids
      any structural mismatch between custom PyTree nodes (like LayerMap) and
      standard dicts.

    • This approach offers explicit control over how transformations are applied
      to leaves, while keeping the logic JAX-compatible and transparent.

    • It is preferred over using :func:`jax.vmap` for this use case, since `vmap`
      is designed for batching along array axes, not for traversing heterogeneous
      pytree leaves with differing shapes or dtypes.

    • The function is side-effect free: the input ``lmap`` is never modified.

    Example
    -------
    >>> # Scale all parameters of diagonal modules by 0.5
    >>> new_lmap = layermap_apply_params_only_flat(
    ...     f=lambda x: 0.5 * x,
    ...     select_idxs=lambda ij: ij[0] == ij[1],
    ...     lmap=lmap,
    ... )

    >>> # Convert upper-triangular module parameters to float32
    >>> new_lmap = layermap_apply_params_only_flat(
    ...     f=lambda x: x.astype(jnp.float32),
    ...     select_idxs=lambda ij: ij[0] < ij[1],
    ...     lmap=lmap,
    ... )

    >>> # Add Gaussian noise to parameters of all modules
    >>> key = jax.random.PRNGKey(0)
    >>> def add_noise(x):
    ...     noise = jax.random.normal(key, shape=x.shape) * 0.01
    ...     return x + noise
    ...
    >>> new_lmap = layermap_apply_params_only_flat(add_noise, lambda ij: True, lmap)

    """
    # 1) Separate numeric (trainable) parameters from statics/state/meta.
    params, statics = eqx.partition(lmap, eqx.is_array)

    # 2) Apply transformation per-module.
    new_params_data = {}
    for i, row in params.row_items():
        new_row = {}
        for j, mod_params in row.items():
            if select_idxs((i, j)):
                # Flatten the module parameters into leaves.
                leaves, treedef = tree_flatten(mod_params)
                # Apply `f` independently to each array leaf.
                new_leaves = [f(x) for x in leaves]
                # Reconstruct the module with the same tree structure.
                new_mod_params = tree_unflatten(treedef, new_leaves)
            else:
                new_mod_params = mod_params
            new_row[j] = new_mod_params
        new_params_data[i] = new_row

    # 3) Re-wrap as a LayerMap if necessary.
    new_params = LayerMap.from_dict(new_params_data)

    # 4) Recombine transformed parameters with untouched statics.
    combined: LayerMap = eqx.combine(new_params, statics)
    return combined
