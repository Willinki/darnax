from typing import Self
from collections.abc import Callable
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from darnax.modules.interfaces import Adapter
from jax import Array
from jax.typing import DTypeLike
from .utils import fetch_tuple_from_arg, pad_2d

KeyArray = Array


class MajorityPooling(Adapter):
    """Implement majority pooling."""

    strength: Array
    stride: int = eqx.field(static=True)
    kernel_size: tuple[int, int] = eqx.field(static=True)
    padding_mode: str | Callable[..., str] | None = eqx.field(static=True)

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        strength: float,
        key: KeyArray,
        stride: int | tuple[int, int] = 1,
        padding_mode: str | Callable[..., str] | None = None,
        dtype: DTypeLike = jnp.float32,
    ):
        """Save properties and init kernel."""
        self.stride = fetch_tuple_from_arg(stride)
        self.kernel_size = fetch_tuple_from_arg(kernel_size)
        self.strength = jnp.asarray(strength)
        self.padding_mode = padding_mode

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Apply majority pooling.

        Pads symmetrically by kh//2, kw//2 so odd kernels preserve H,W when stride=1.
        """
        # normalize params
        kh, kw = self.kernel_size
        stride = fetch_tuple_from_arg(self.stride)

        # symmetric half-kernel padding (not full kernel!)
        pad_h = kh // 2
        pad_w = kw // 2
        x_pad = pad_2d(x, pad_h, pad_w, self.padding_mode)

        n, h_in, w_in, c_in = x_pad.shape

        # Result: (N, Ho, Wo, Kh * Kw * Cin)
        patches = lax.conv_general_dilated_patches(
            x_pad,
            filter_shape=self.kernel_size,
            window_strides=stride,
            padding="VALID",
            lhs_dilation=(1, 1),
            rhs_dilation=(1, 1),
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        # Result: (N, Ho, Wo, Kh, Kw, Cin)
        patches = patches.reshape(*patches.shape[:-1], kh, kw, c_in)

        # Sum over spatial kernel dims -> (N, Ho, Wo, Cin)
        sums = jnp.sum(patches, axis=(-3, -2))
        # majority sign: >0 -> 1 else -1 (ties -> -1)
        majority = jnp.where(sums > 0, 1, -1)
        return self.strength * majority

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Update parameters using conv backward."""
        # nothing to update
        zero_update = jax.tree.map(jnp.zeros_like, self, is_leaf=eqx.is_inexact_array)
        return zero_update


class ConstantUnpooling(Adapter):
    """Increases the size of each pixel by a constant factor.

    Basically a pixel becomes a cube of constant elements.
    Optionally rescaled by strength.
    """

    strength: Array
    kernel_size: tuple[int, int] = eqx.field(static=True)
    unpad: tuple[int, int] | None = eqx.field(static=True)

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        strength: float,
        dtype: DTypeLike = jnp.float32,
        unpad: int | tuple[int, int] | None = None,
    ):
        """Save properties and init kernel.

        Args:
            kernel_size: Expansion factor per dimension
            strength: Multiplicative scaling factor
            dtype: Data type for strength
            unpad: Amount to crop from each side (symmetric). If None, no cropping.
                   Useful for matching pooling input shape with symmetric padding.

        """
        self.kernel_size = fetch_tuple_from_arg(kernel_size)
        self.strength = jnp.asarray(strength, dtype=dtype)
        self.unpad = fetch_tuple_from_arg(unpad) if unpad is not None else None

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Implement constant unpooling with optional unpadding."""
        assert len(x.shape) == 4
        increased = jnp.repeat(
            jnp.repeat(x, repeats=self.kernel_size[0], axis=-3),
            repeats=self.kernel_size[1],
            axis=-2,
        )

        if self.unpad is not None:
            pad_h, pad_w = self.unpad
            if pad_h > 0:
                increased = increased[:, pad_h:-pad_h, :, :]
            if pad_w > 0:
                increased = increased[:, :, pad_w:-pad_w, :]

        return self.strength * increased

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Update parameters using conv backward."""
        # nothing to update
        zero_update = jax.tree.map(jnp.zeros_like, self, is_leaf=eqx.is_inexact_array)
        return zero_update


class GlobalMajorityPooling(Adapter):
    """Implement majority pooling. The pooled dimension is compressed."""

    strength: Array
    axis: int | tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        strength: float,
        axis: int | tuple[int, ...],
        dtype: DTypeLike = jnp.float32,
    ):
        """Save properties and init kernel."""
        self.strength = jnp.asarray(strength, dtype=dtype)
        self.axis = int(axis) if isinstance(axis, int) else tuple(axis)

    def __call__(self, x: Array, rng=None) -> Array:
        """Ingest a 4 dimensional array and returns the majority along axis.

        Dimensions are [B, h, W, C]
        """
        return self.strength * jnp.where(
            jnp.sum(
                x,
                axis=self.axis,
                keepdims=False,
            )
            > 0,
            1,
            -1,
        )

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Update parameters using conv backward."""
        # nothing to update
        zero_update = jax.tree.map(jnp.zeros_like, self, is_leaf=eqx.is_inexact_array)
        return zero_update


class GlobalUnpooling(Adapter):
    """Implement global majority unpooling. The axis dimension is inserted."""

    strength: Array
    axis: int | tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        strength: float,
        axis: int | tuple[int, ...],
        dtype: DTypeLike = jnp.float32,
    ):
        """Save properties and init kernel."""
        self.strength = jnp.asarray(strength, dtype=dtype)
        self.axis = int(axis) if isinstance(axis, int) else tuple(axis)

    def __call__(self, x: Array, rng=None) -> Array:
        """Ingest a n-dimensional array. Expands along axis and with constant value.

        Value is taken for following dimensions.

        """
        return self.strength * jnp.expand_dims(x, axis=self.axis)

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Update parameters using conv backward."""
        # nothing to update
        zero_update = jax.tree.map(jnp.zeros_like, self, is_leaf=eqx.is_inexact_array)
        return zero_update
