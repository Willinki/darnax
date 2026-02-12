"""Grouped convolution forward/backward helpers with proper variance normalization.

This module provides convolution operations for NHWC (batch, height, width, channels)
input layout and HWIO (height, width, input_channels, output_channels) kernel layout.

Variance Normalization Principles
----------------------------------
Forward Pass:
    Kernel initialization with Var(W) = 1/fan_in ensures that if the input has
    unit variance, the output will also have unit variance (before any scaling factor).

    fan_in = Kh × Kw × Cin_per_group

Backward Pass:
    Gradient normalization by 1/sqrt(N × Ho × Wo) ensures that if both the input
    and output gradient have unit variance, the kernel gradient will have unit variance.

    CRITICAL: The normalization is over spatial and batch dimensions only.
    The fan_in term does NOT appear in backward normalization.

Grouping:
    All functions support grouped convolutions with contiguous channel partitioning.
    For groups > 1, channels are divided into contiguous blocks.

Functions
---------
conv_forward : Standard 2D convolution
conv_transpose_forward : Transposed convolution via rhs_dilation
conv_backward : Kernel gradient accumulator (variance-normalized)
conv_backward_with_threshold : Gated kernel gradient accumulator
conv_transpose_backward_with_threshold : Transposed geometry with gating

Notes
-----
All backward-like functions return variance-normalized gradients scaled by:
    scale = 1 / sqrt(N × Ho × Wo)
where N is batch size, Ho and Wo are output spatial dimensions.

"""

from collections.abc import Callable
from jax import lax
import jax.numpy as jnp

Array = jnp.ndarray
IntPair = int | tuple[int, int]


# ---------- helpers ----------
def fetch_tuple_from_arg(x: IntPair) -> tuple[int, int]:
    """Normalize a stride/dilation argument to a (sh, sw) tuple.

    Parameters
    ----------
    x : int or tuple[int, int]
        Stride or dilation value. If int, applies same value to both dimensions.
        If tuple or list, must be length 2.

    Returns
    -------
    tuple[int, int]
        Normalized (height, width) tuple.

    Raises
    ------
    ValueError
        If x is not an int or length-2 sequence.

    """
    if isinstance(x, int):
        return (x, x)
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return (int(x[0]), int(x[1]))
    raise ValueError("Expected int or length-2 sequence for stride/dilation.")


def pad_2d(x: Array, pad_h: int, pad_w: int, mode: str | Callable[..., str] | None) -> Array:
    """Apply 2D spatial padding to an NHWC tensor.

    Parameters
    ----------
    x : Array
        Input tensor of shape (N, H, W, C).
    pad_h : int
        Padding applied to height dimension (top and bottom).
    pad_w : int
        Padding applied to width dimension (left and right).
    mode : str, callable, or None
        Padding mode passed to jax.numpy.pad.
        If None, no padding is applied and input is returned unchanged.

    Returns
    -------
    Array
        Padded tensor of shape (N, H + 2*pad_h, W + 2*pad_w, C).

    """
    if mode is None:
        return x
    return jnp.pad(
        x,
        pad_width=[
            (0, 0),
            (pad_h, pad_h),
            (pad_w, pad_w),
            (0, 0),
        ],
        mode=mode,
    )


# ---------- forward passes ----------
def conv_forward(
    x: Array,
    kernel: Array,
    stride: IntPair = 1,
    padding_mode: str | Callable[..., str] | None = None,
) -> Array:
    """Forward pass of a 2D convolution.

    Parameters
    ----------
    x : Array
        Input tensor of shape (N, H, W, Cin).
    kernel : Array
        Convolution kernel of shape (Kh, Kw, Cin, Cout).
    stride : int or tuple[int, int], default=1
        Stride for the convolution.
    padding_mode : str, callable, or None, default=None
        Padding mode to use via pad_2d.

    Returns
    -------
    Array
        Output tensor of shape (N, Ho, Wo, Cout).

    """
    stride = fetch_tuple_from_arg(stride)
    kh, kw = kernel.shape[0], kernel.shape[1]
    x_pad = pad_2d(x, kh // 2, kw // 2, padding_mode)
    y = lax.conv_general_dilated(
        lhs=x_pad,
        rhs=kernel,
        window_strides=stride,
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y


def conv_transpose_forward(
    x: Array,
    kernel: Array,
    stride: IntPair = 1,
    padding_mode: str | Callable[..., str] | None = None,
) -> Array:
    """
    Forward pass of a 2D transposed convolution.

    Parameters
    ----------
    x : Array
        Input tensor of shape (N, H, W, Cin).
    kernel : Array
        Kernel of shape (Kh, Kw, Cin, Cout).
    stride : int or tuple[int, int], default=1
        Upsampling factor implemented via rhs_dilation.
    padding_mode : str, callable, or None, default=None
        Padding mode for pad_2d.

    Returns
    -------
    Array
        Output tensor of shape (N, Ho, Wo, Cout).

    """
    stride = fetch_tuple_from_arg(stride)
    kh, kw, c_in_k, c_out = kernel.shape
    assert c_in_k == x.shape[-1], "kernel in-channels mismatch"

    x_pad = pad_2d(x, kh // 2, kw // 2, padding_mode)

    y = lax.conv_general_dilated(
        lhs=x_pad,
        rhs=kernel,
        window_strides=(1, 1),
        padding=((kh - 1, kh - 1), (kw - 1, kw - 1)),
        lhs_dilation=(1, 1),
        rhs_dilation=stride,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y


# ---------- backward-like accumulators (kernel gradients) ----------
def _expand_grouped_to_full(
    accum_g: Array,
    groups: int,
    Cin_per: int,
    Cout_per: int,
    kh: int,
    kw: int,
) -> Array:
    """Expand grouped accumulator to full block-diagonal kernel gradient.

    Converts (Kh, Kw, G, Cin_per, Cout_per) to (Kh, Kw, Cin, Cout) with
    block-diagonal structure (cross-group entries are zero).

    Fully jittable — no Python loops.

    Parameters
    ----------
    accum_g : Array
        Grouped accumulator of shape (Kh, Kw, G, Cin_per, Cout_per).
    groups : int
        Number of groups.
    Cin_per : int
        Input channels per group.
    Cout_per : int
        Output channels per group.
    kh : int
        Kernel height.
    kw : int
        Kernel width.

    Returns
    -------
    Array
        Block-diagonal kernel gradient of shape (Kh, Kw, G*Cin_per, G*Cout_per).

    """
    eye_g = jnp.eye(groups, dtype=accum_g.dtype)  # (G, G)
    # (Kh, Kw, G, Cin_per, Cout_per) x (G, G) -> (Kh, Kw, G_in, Cin_per, G_out, Cout_per)
    # nonzero only when G_in == G_out
    accum_bd = jnp.einsum("kwgco, gj -> kwgcjo", accum_g, eye_g)
    # merge (G_in, Cin_per) -> Cin and (G_out, Cout_per) -> Cout
    return accum_bd.reshape(kh, kw, groups * Cin_per, groups * Cout_per)


def conv_backward(
    x: Array,
    y: Array,
    kernel_shape: tuple[int, int],
    groups: int = 1,
    strides: IntPair = (1, 1),
    padding_mode: str | Callable[..., str] | None = None,
    lhs_dilation: IntPair = (1, 1),
    rhs_dilation: IntPair = (1, 1),
) -> Array:
    """Compute variance-normalized kernel gradient for 2D convolution.

    Parameters
    ----------
    x : Array
        Input tensor of shape (N, H, W, Cin).
    y : Array
        Gradient w.r.t. convolution output, shape (N, Ho, Wo, Cout).
    kernel_shape : tuple[int, int]
        Spatial size of kernel (Kh, Kw).
    groups : int, default=1
        Number of groups for grouped convolution.
    strides : int or tuple[int, int], default=(1, 1)
        Convolution stride.
    padding_mode : str, callable, or None, default=None
        Padding mode for patch extraction.
    lhs_dilation : int or tuple[int, int], default=(1, 1)
        Left-hand-side dilation.
    rhs_dilation : int or tuple[int, int], default=(1, 1)
        Right-hand-side dilation.

    Returns
    -------
    Array
        Kernel gradient of shape (Kh, Kw, Cin, Cout), normalized by
        1 / sqrt(N × Ho × Wo).

    """
    n, ho, wo, Cout = y.shape
    kh, kw = kernel_shape
    Cin = x.shape[-1]
    strides = fetch_tuple_from_arg(strides)

    pad_h = kh // 2
    pad_w = kw // 2
    x_pad = pad_2d(x, pad_h, pad_w, padding_mode)

    patches = lax.conv_general_dilated_patches(
        x_pad,
        filter_shape=(kh, kw),
        window_strides=strides,
        padding="VALID",
        lhs_dilation=fetch_tuple_from_arg(lhs_dilation),
        rhs_dilation=fetch_tuple_from_arg(rhs_dilation),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    n, ho, wo, _ = patches.shape
    patches = patches.reshape(n, ho, wo, Cin, kh, kw)  # interpret packed axis as (Cin, Kh, Kw)
    patches = jnp.transpose(patches, (0, 1, 2, 4, 5, 3))  # -> (N, Ho, Wo, Kh, Kw, Cin)

    if groups == 1:
        accum = jnp.einsum("nhwklc, nhwo -> klco", patches, y)
        scale = 1.0 / jnp.sqrt(jnp.asarray(n * ho * wo, dtype=accum.dtype))
        return accum * scale

    assert Cin % groups == 0 and Cout % groups == 0, "Cin and Cout must be divisible by groups"
    Cin_per = Cin // groups
    Cout_per = Cout // groups

    patches_g = patches.reshape(n, ho, wo, kh, kw, groups, Cin_per)
    out_g = y.reshape(n, ho, wo, groups, Cout_per)

    accum_g = jnp.einsum("nhwklgc, nhwgo -> klgco", patches_g, out_g)
    scale = 1.0 / jnp.sqrt(jnp.asarray(n * ho * wo, dtype=accum_g.dtype))
    accum_g = accum_g * scale

    return _expand_grouped_to_full(accum_g, groups, Cin_per, Cout_per, kh, kw)


def conv_backward_with_threshold(
    x: Array,
    y: Array,
    y_hat: Array,
    threshold: Array,
    kernel_shape: tuple[int, int],
    groups: int = 1,
    strides: IntPair = (1, 1),
    padding_mode: str | Callable[..., str] | None = None,
    lhs_dilation: IntPair = (1, 1),
    rhs_dilation: IntPair = (1, 1),
) -> Array:
    """Kernel gradient accumulator with elementwise threshold gating.

    Only accumulates gradient at positions where (y * y_hat) < threshold.

    Parameters
    ----------
    x : Array
        Input tensor of shape (N, H, W, Cin).
    y : Array
        Primary signal at convolution output, shape (N, Ho, Wo, Cout).
    y_hat : Array
        Secondary signal used for gating, shape (N, Ho, Wo, Cout).
    threshold : scalar Array
        Gating threshold.
    kernel_shape : tuple[int, int]
        Spatial kernel size (Kh, Kw).
    groups : int, default=1
        Number of contiguous groups.
    strides : int or tuple[int, int], default=(1, 1)
        Convolution stride.
    padding_mode : str, callable, or None, default=None
        Padding mode for patch extraction.
    lhs_dilation : int or tuple[int, int], default=(1, 1)
        Left-hand-side dilation.
    rhs_dilation : int or tuple[int, int], default=(1, 1)
        Right-hand-side dilation.

    Returns
    -------
    Array
        Gated, variance-normalized kernel gradient of shape (Kh, Kw, Cin, Cout).

    """
    n, ho, wo, Cout = y.shape
    kh, kw = kernel_shape
    Cin = x.shape[-1]
    strides = fetch_tuple_from_arg(strides)

    pad_h = kh // 2
    pad_w = kw // 2
    x_pad = pad_2d(x, pad_h, pad_w, padding_mode)

    patches = lax.conv_general_dilated_patches(
        x_pad,
        filter_shape=(kh, kw),
        window_strides=strides,
        padding="VALID",
        lhs_dilation=fetch_tuple_from_arg(lhs_dilation),
        rhs_dilation=fetch_tuple_from_arg(rhs_dilation),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    n, ho, wo, _ = patches.shape
    patches = patches.reshape(n, ho, wo, Cin, kh, kw)  # interpret packed axis as (Cin, Kh, Kw)
    patches = jnp.transpose(patches, (0, 1, 2, 4, 5, 3))  # -> (N, Ho, Wo, Kh, Kw, Cin)

    # Apply threshold gating
    mask = (y * y_hat < threshold).astype(y.dtype)
    out_masked = y * mask

    if groups == 1:
        accum = jnp.einsum("nhwklc, nhwo -> klco", patches, out_masked)
        scale = 1.0 / jnp.sqrt(jnp.asarray(n * ho * wo, dtype=accum.dtype))
        return accum * scale

    assert Cin % groups == 0 and Cout % groups == 0, "Cin and Cout must be divisible by groups"
    Cin_per = Cin // groups
    Cout_per = Cout // groups

    patches_g = patches.reshape(n, ho, wo, kh, kw, groups, Cin_per)
    out_g = out_masked.reshape(n, ho, wo, groups, Cout_per)

    accum_g = jnp.einsum("nhwklgc, nhwgo -> klgco", patches_g, out_g)
    scale = 1.0 / jnp.sqrt(jnp.asarray(n * ho * wo, dtype=accum_g.dtype))
    accum_g = accum_g * scale

    return _expand_grouped_to_full(accum_g, groups, Cin_per, Cout_per, kh, kw)


def conv_transpose_backward_with_threshold(
    x: Array,
    y: Array,
    y_hat: Array,
    threshold: Array,
    kernel_shape: tuple[int, int],
    stride: IntPair,
    groups: int = 1,
    padding_mode: str | Callable[..., str] | None = None,
) -> Array:
    """Kernel gradient for transposed convolution with threshold gating.

    Matches the geometry of conv_transpose_forward (uses rhs_dilation=stride).

    Parameters
    ----------
    x : Array
        Input tensor of shape (N, H, W, Cin).
    y : Array
        Primary signal at transposed conv output, shape (N, Ho, Wo, Cout).
    y_hat : Array
        Secondary signal for gating, shape (N, Ho, Wo, Cout).
    threshold : scalar Array
        Gating threshold.
    kernel_shape : tuple[int, int]
        Spatial kernel shape (Kh, Kw).
    stride : int or tuple[int, int]
        Upsampling factor used in forward transposed convolution.
    groups : int, default=1
        Number of contiguous groups.
    padding_mode : str, callable, or None, default=None
        Padding mode for pad_2d.

    Returns
    -------
    Array
        Kernel gradient of shape (Kh, Kw, Cin, Cout), variance-normalized.

    """
    n, ho, wo, Cout = y.shape
    kh, kw = kernel_shape
    Cin = x.shape[-1]

    x_pad = pad_2d(x, kh // 2, kw // 2, padding_mode)

    patches = lax.conv_general_dilated_patches(
        x_pad,
        filter_shape=(kh, kw),
        window_strides=(1, 1),
        padding=((kh - 1, kh - 1), (kw - 1, kw - 1)),
        lhs_dilation=(1, 1),
        rhs_dilation=fetch_tuple_from_arg(stride),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    n, ho, wo, _ = patches.shape
    patches = patches.reshape(n, ho, wo, Cin, kh, kw)  # interpret packed axis as (Cin, Kh, Kw)
    patches = jnp.transpose(patches, (0, 1, 2, 4, 5, 3))  # -> (N, Ho, Wo, Kh, Kw, Cin)

    # Apply threshold gating
    mask = (y * y_hat < threshold).astype(patches.dtype)
    out_masked = y * mask

    if groups == 1:
        accum = jnp.einsum("nhwklc, nhwo -> klco", patches, out_masked)
        scale = 1.0 / jnp.sqrt(jnp.asarray(n * ho * wo, dtype=accum.dtype))
        return accum * scale

    assert Cin % groups == 0 and Cout % groups == 0, "Cin and Cout must be divisible by groups"
    Cin_per = Cin // groups
    Cout_per = Cout // groups

    patches_g = patches.reshape(n, ho, wo, kh, kw, groups, Cin_per)
    out_g = out_masked.reshape(n, ho, wo, groups, Cout_per)

    accum_g = jnp.einsum("nhwklgc, nhwgo -> klgco", patches_g, out_g)
    scale = 1.0 / jnp.sqrt(jnp.asarray(n * ho * wo, dtype=accum_g.dtype))
    accum_g = accum_g * scale

    return _expand_grouped_to_full(accum_g, groups, Cin_per, Cout_per, kh, kw)
