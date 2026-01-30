import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from darnax.modules.conv.pooling import MajorityPooling, ConstantUnpooling


def _rand(key, shape, dtype=jnp.float32):
    return jax.random.normal(key, shape, dtype=dtype)


def test_majority_pooling_basic_behavior():
    # Build an input where a 2x2 window majority is positive, negative, and balanced.
    # Batch=1, H=4, W=4, C=1
    x = jnp.array(
        [
            [
                [[1.0], [1.0], [-1.0], [-1.0]],
                [[1.0], [1.0], [-1.0], [-1.0]],
                [[-1.0], [-1.0], [1.0], [1.0]],
                [[-1.0], [-1.0], [1.0], [1.0]],
            ]
        ],
        dtype=jnp.float32,
    )  # shape (1,4,4,1)

    mp = MajorityPooling(
        kernel_size=2, strength=2.0, key=jax.random.PRNGKey(0), stride=2, padding_mode=None
    )

    out = mp(x)  # stride=2, so output spatial dims should be 2x2
    assert out.shape == (1, 2, 2, 1)

    # top-left 2x2 window has sum +4 -> majority positive -> +1 scaled by strength
    assert jnp.allclose(out[0, 0, 0, 0], 2.0)
    # top-right 2x2 window has sum -4 -> majority negative -> -2.0
    assert jnp.allclose(out[0, 0, 1, 0], -2.0)
    # bottom-left negative
    assert jnp.allclose(out[0, 1, 0, 0], -2.0)
    # bottom-right positive
    assert jnp.allclose(out[0, 1, 1, 0], 2.0)


def test_majority_pooling_threshold_and_padding():
    x = jnp.zeros((1, 3, 3, 1), dtype=jnp.float32)
    x = x.at[0, 1, 1, 0].set(1.0)

    mp = MajorityPooling(
        kernel_size=3, strength=1.5, key=jax.random.PRNGKey(1), stride=1, padding_mode="constant"
    )
    out = mp(x)
    assert out.shape == (1, 3, 3, 1)

    # center should be positive scaled
    assert jnp.allclose(out[0, 1, 1, 0], 1.5)

    # NOTE: with symmetric kh//2 padding, the center element is included in every 3x3 patch
    # extracted for stride=1, so the corners will also be positive.
    assert jnp.allclose(out[0, 0, 0, 0], 1.5)
    assert jnp.allclose(out[0, 0, 2, 0], 1.5)
    assert jnp.allclose(out[0, 2, 0, 0], 1.5)
    assert jnp.allclose(out[0, 2, 2, 0], 1.5)


def test_majority_pooling_backward_zero_update_and_equinox_filter():
    # Create a module and call backward; the returned tree should contain same pytree
    # structure and have only zero arrays (since there's nothing to update).
    mp = MajorityPooling(
        kernel_size=2, strength=0.5, key=jax.random.PRNGKey(2), stride=1, padding_mode=None
    )

    # small dummy inputs
    x = jnp.ones((1, 4, 4, 1), dtype=jnp.float32)
    y = mp(x)
    y_hat = jnp.zeros_like(y)

    update = mp.backward(x, y, y_hat, gate=None)

    # update must be an Equinox pytree of the same type
    assert isinstance(update, MajorityPooling)

    # All array leaves in update should be zeros. Use eqx.filter to get arrays.
    res = eqx.filter(update, eqx.is_inexact_array)
    # eqx.filter may return either (matched, rest) or just matched depending on Equinox version
    if isinstance(res, tuple) and len(res) == 2:
        arrays, rest = res
    else:
        arrays = res

    # use jax.tree_util.tree_leaves to get leaves (works regardless of eqx version)
    leaves = jtu.tree_leaves(arrays)
    assert leaves, "expected at least one array leaf in the update object"
    for leaf in leaves:
        # leaf can be jax arrays; compare to zero
        assert jnp.allclose(leaf, 0.0)


def test_majority_pooling_jittable_call_and_no_grad():
    mp = MajorityPooling(
        kernel_size=2, strength=1.0, key=jax.random.PRNGKey(3), stride=1, padding_mode=None
    )
    x = jnp.ones((1, 4, 4, 1), dtype=jnp.float32)

    # JIT the module-call; pass module as explicit argument so jit sees it as a pytree input.
    jit_call = jax.jit(lambda module, inp: module(inp))
    out = jit_call(mp, x)
    # kernel_size=2, padding_mode=None -> output is (H - 2 + 1, W - 2 + 1) = (3,3)
    assert out.shape == (1, 3, 3, 1)

    # Ensure no gradient is attempted/needed; we just ensure jax.grad on the module call fails or is not used.
    # The module is not differentiable by design (no float parameters to learn here), but we assert grad on outputs wrt input exists.
    g = jax.grad(lambda inp: jnp.sum(jit_call(mp, inp)))(x)
    assert g.shape == x.shape


def test_constant_unpooling_repeat_and_jit_and_backward():
    # Input shape (1,2,2,1), kernel_size=(2,3) -> output (1,4,6,1)
    cu = ConstantUnpooling(kernel_size=(2, 3), strength=0.25)
    x = jnp.arange(4, dtype=jnp.float32).reshape(1, 2, 2, 1)  # values 0..3
    out = cu(x)
    assert out.shape == (1, 4, 6, 1)

    s = float(cu.strength)

    # check repeated block: out[0,0:2,0:3,0] should be filled with x[0,0,0,0] == 0 scaled by strength
    assert jnp.allclose(out[0, 0:2, 0:3, 0], s * 0.0)

    # check next block corresponds to x[0,0,1,0] == 1 scaled by strength
    assert jnp.allclose(out[0, 0:2, 3:6, 0], s * 1.0)

    # also verify lower row blocks correspond to x[0,1,0,0] == 2 and x[0,1,1,0] == 3
    assert jnp.allclose(out[0, 2:4, 0:3, 0], s * 2.0)
    assert jnp.allclose(out[0, 2:4, 3:6, 0], s * 3.0)

    # JIT the call (module passed as pytree)
    jit_call = jax.jit(lambda module, inp: module(inp))
    out_jit = jit_call(cu, x)
    assert jnp.allclose(out_jit, out)

    # backward returns zero-update structure
    upd = cu.backward(x, out, out)
    import jax.tree_util as jtu

    leaves = jtu.tree_leaves(upd)
    array_leaves = [leaf for leaf in leaves if eqx.is_inexact_array(leaf)]
    for leaf in array_leaves:
        assert jnp.allclose(leaf, 0.0)
