"""Pooling and unpooling adapters for neural networks.

This module provides majority pooling and constant unpooling operations,
both local and global variants.
"""

from collections.abc import Callable
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.typing import DTypeLike

from darnax.modules.interfaces import Adapter

from .utils import fetch_tuple_from_arg, pad_2d

KeyArray = Array


class MajorityPooling(Adapter):
    """Implement majority pooling over spatial dimensions.

    Applies a sliding window over spatial dimensions and outputs the majority
    sign within each window, scaled by a strength parameter.

    Attributes:
        strength: Multiplicative scaling factor for output.
        stride: Step sizes for the sliding window.
        kernel_size: Height and width of the pooling window.
        padding_mode: Padding strategy (e.g., "edge", "constant") or callable.

    """

    strength: Array
    stride: tuple[int, int] = eqx.field(static=True)
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
        """Initialize majority pooling layer.

        Args:
            kernel_size: Size of the pooling window (height, width).
            strength: Multiplicative factor applied to the output.
            key: Random key (unused, kept for interface compatibility).
            stride: Step size for the sliding window. Default is 1.
            padding_mode: Padding strategy or callable. If None, no padding.
            dtype: Data type for the strength parameter. Default is float32.

        """
        self.stride = fetch_tuple_from_arg(stride)
        self.kernel_size = fetch_tuple_from_arg(kernel_size)
        self.strength = jnp.asarray(strength)
        self.padding_mode = padding_mode

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Apply majority pooling to input.

        Pads symmetrically by kh//2, kw//2 so odd kernels preserve H, W when
        stride=1.

        Args:
            x: Input array of shape (N, H, W, C).
            rng: Random key (unused).

        Returns:
            Pooled array with shape (N, H_out, W_out, C) where each value is
            the majority sign within the pooling window, scaled by strength.

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
        """Compute parameter updates (no-op for this layer).

        Args:
            x: Input array.
            y: Target array.
            y_hat: Predicted array.
            gate: Optional gating array.

        Returns:
            Zero updates for all parameters.

        """
        # nothing to update
        zero_update: Self = jax.tree.map(jnp.zeros_like, self, is_leaf=eqx.is_inexact_array)
        return zero_update


class ConstantUnpooling(Adapter):
    """Increase spatial dimensions by repeating pixels.

    Each pixel is expanded into a constant-valued block, optionally scaled by
    a strength parameter. Can optionally crop edges to match original input shape.

    Attributes:
        strength: Multiplicative scaling factor for output.
        kernel_size: Expansion factor per spatial dimension (height, width).
        unpad: Amount to crop from each side after expansion. If None, no cropping.

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
        """Initialize constant unpooling layer.

        Args:
            kernel_size: Expansion factor per dimension (height, width).
            strength: Multiplicative scaling factor applied to output.
            dtype: Data type for the strength parameter. Default is float32.
            unpad: Amount to crop from each side (symmetric). If None, no cropping.
                Useful for matching pooling input shape with symmetric padding.

        """
        self.kernel_size = fetch_tuple_from_arg(kernel_size)
        self.strength = jnp.asarray(strength, dtype=dtype)
        self.unpad = fetch_tuple_from_arg(unpad) if unpad is not None else None

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Apply constant unpooling to input.

        Args:
            x: Input array of shape (N, H, W, C).
            rng: Random key (unused).

        Returns:
            Expanded array of shape (N, H', W', C) where H' and W' are
            determined by kernel_size and unpad parameters.

        Raises:
            AssertionError: If input is not 4-dimensional.

        """
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
        """Compute parameter updates (no-op for this layer).

        Args:
            x: Input array.
            y: Target array.
            y_hat: Predicted array.
            gate: Optional gating array.

        Returns:
            Zero updates for all parameters.

        """
        # nothing to update
        zero_update: Self = jax.tree.map(jnp.zeros_like, self, is_leaf=eqx.is_inexact_array)
        return zero_update


class GlobalMajorityPooling(Adapter):
    """Apply majority pooling along specified axes.

    Computes the majority sign by summing along the specified axes and
    determining if the result is positive or negative.

    Attributes:
        strength: Multiplicative scaling factor for output.
        axis: Axis or axes along which to compute majority.

    """

    strength: Array
    axis: int | tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        strength: float,
        axis: int | tuple[int, ...],
        dtype: DTypeLike = jnp.float32,
    ):
        """Initialize global majority pooling layer.

        Args:
            strength: Multiplicative factor applied to the output.
            axis: Axis or axes along which to compute majority pooling.
            dtype: Data type for the strength parameter. Default is float32.

        """
        self.strength = jnp.asarray(strength, dtype=dtype)
        self.axis = int(axis) if isinstance(axis, int) else tuple(axis)

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Apply global majority pooling to input.

        Args:
            x: Input array of shape (B, H, W, C) or any n-dimensional shape.
            rng: Random key (unused).

        Returns:
            Array with specified axes reduced, containing majority signs
            (1 or -1) scaled by strength.

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
        """Compute parameter updates (no-op for this layer).

        Args:
            x: Input array.
            y: Target array.
            y_hat: Predicted array.
            gate: Optional gating array.

        Returns:
            Zero updates for all parameters.

        """
        # nothing to update
        zero_update: Self = jax.tree.map(jnp.zeros_like, self, is_leaf=eqx.is_inexact_array)
        return zero_update


class GlobalUnpooling(Adapter):
    """Expand array by inserting new axes.

    Adds singleton dimensions along specified axes and scales by strength.

    Attributes:
        strength: Multiplicative scaling factor for output.
        axis: Axis or axes along which to insert new dimensions.

    """

    strength: Array
    axis: int | tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        strength: float,
        axis: int | tuple[int, ...],
        dtype: DTypeLike = jnp.float32,
    ):
        """Initialize global unpooling layer.

        Args:
            strength: Multiplicative factor applied to the output.
            axis: Axis or axes along which to insert new dimensions.
            dtype: Data type for the strength parameter. Default is float32.

        """
        self.strength = jnp.asarray(strength, dtype=dtype)
        self.axis = int(axis) if isinstance(axis, int) else tuple(axis)

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Apply global unpooling to input.

        Args:
            x: Input n-dimensional array.
            rng: Random key (unused).

        Returns:
            Array with new singleton dimensions inserted at specified axes,
            scaled by strength.

        """
        return self.strength * jnp.expand_dims(x, axis=self.axis)

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Compute parameter updates (no-op for this layer).

        Args:
            x: Input array.
            y: Target array.
            y_hat: Predicted array.
            gate: Optional gating array.

        Returns:
            Zero updates for all parameters.

        """
        # nothing to update
        zero_update: Self = jax.tree.map(jnp.zeros_like, self, is_leaf=eqx.is_inexact_array)
        return zero_update
