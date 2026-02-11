"""
conv_utils.py

Grouped convolution forward/backward helpers (NHWC / HWIO layouts).

- conv_forward: standard padded conv (NHWC input, HWIO kernel, NHWC output)
- conv_transpose_forward: transposed conv via rhs_dilation geometry
- conv_backward: kernel gradient-like accumulator (groupwise, contiguous)
- conv_backward_with_threshold: same as conv_backward but gates out_grad by threshold
- conv_transpose_backward_with_threshold: transposed geometry counterpart

All backward-like functions apply variance-normalizing scale:
    scale = 1 / sqrt( (N * Ho * Wo) * fan_in_per )
where fan_in_per = (Cin_per_group * Kh * Kw).

Grouping assumes contiguous partitioning of channels.
"""

from collections.abc import Callable
from jax import lax
import jax.numpy as jnp

Array = jnp.ndarray
IntPair = int | tuple[int, int]


# ---------- helpers ----------
def fetch_tuple_from_arg(x: IntPair) -> tuple[int, int]:
    """
    Normalize a stride/dilation argument to a (sh, sw) tuple.

    Accepts an int, a 2-tuple, or any length-2 sequence.
    """
    if isinstance(x, int):
        return (x, x)
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return (int(x[0]), int(x[1]))
    raise ValueError("Expected int or length-2 sequence for stride/dilation.")


def pad_2d(x: Array, pad_h: int, pad_w: int, mode: str | Callable[..., str] | None):
    """
    Apply 2D spatial padding to an NHWC tensor.

    Pads the height and width dimensions symmetrically while leaving
    batch and channel dimensions unchanged.

    Parameters
    ----------
    x : Array
        Input tensor of shape (N, H, W, C).
    pad_h : int
        Padding applied to the height dimension (top and bottom).
    pad_w : int
        Padding applied to the width dimension (left and right).
    mode : str or callable or None
        Padding mode passed to jax.numpy.pad. If None, no padding is applied.

    Returns
    -------
    Array
        Padded tensor of shape (N, H + 2*pad_h, W + 2*pad_w, C),
        or the original tensor if mode is None.

    """
    if mode is None:
        return x
    return jnp.pad(
        x,
        pad_width=[
            (0, 0),  # batch
            (pad_h, pad_h),  # height
            (pad_w, pad_w),  # width
            (0, 0),  # channels
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
    """
    Forward pass of a 2D convolution (NHWC input, HWIO kernel, NHWC output).

    Applies symmetric spatial padding and uses lax.conv_general_dilated.

    Parameters
    ----------
    x : Array
        Input tensor of shape (N, H, W, Cin).
    kernel : Array
        Convolution kernel of shape (Kh, Kw, Cin, Cout).
    stride : int or tuple[int, int]
        Stride (int -> same for both dims).
    padding_mode : str/callable/None
        Padding mode to use via pad_2d. If None, no padding applied.

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
    Forward pass of a 2D transposed convolution (NHWC / HWIO layout).

    Uses rhs_dilation=stride to emulate conv-transpose while preserving
    NHWC ordering. Geometry chosen to be mirrored by the corresponding
    backward-with-threshold function.

    Parameters
    ----------
    x : Array
        Input tensor of shape (N, H, W, Cin).
    kernel : Array
        Kernel of shape (Kh, Kw, Cin, Cout).
    stride : int or tuple[int, int]
        Upsampling factor implemented via rhs_dilation.
    padding_mode : str/callable/None
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
def conv_backward(
    x: Array,
    out_grad: Array,
    kernel_shape: tuple[int, int],
    groups: int = 1,
    strides: IntPair = (1, 1),
    padding_mode: str | Callable[..., str] | None = None,
    lhs_dilation: IntPair = (1, 1),
    rhs_dilation: IntPair = (1, 1),
) -> Array:
    """
    Compute a kernel-gradient-like accumulator for a 2D convolution.

    Algebraically equivalent to the true kernel gradient for a grouped
    convolution (contiguous grouping) except that the returned accumulator
    is rescaled to control variance:

        scale = 1 / sqrt( (N * Ho * Wo) * fan_in_per )

    where fan_in_per = (Cin_per_group * Kh * Kw).

    Parameters
    ----------
    x : Array
        Input tensor of shape (N, H, W, Cin).
    out_grad : Array
        Gradient w.r.t. conv output of shape (N, Ho, Wo, Cout).
    kernel_shape : (Kh, Kw)
        Spatial size of the kernel.
    groups : int
        Number of groups (contiguous partition of Cin and Cout). Default 1.
    strides, padding_mode, lhs_dilation, rhs_dilation :
        Passed through to patch extraction (geometry must match forward).

    Returns
    -------
    Array
        Kernel accumulator of shape (Kh, Kw, Cin, Cout), variance-normalized.

    """
    n, ho, wo, Cout = out_grad.shape
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
    # (N, Hp, Wp, Kh*Kw*Cin) -> (N, Hp, Wp, Kh, Kw, Cin)
    patches = patches.reshape(patches.shape[0], patches.shape[1], patches.shape[2], kh, kw, Cin)

    if groups == 1:
        accum = jnp.einsum("n h w k l c, n h w o -> k l c o", patches, out_grad)
        fan_in_per = Cin * kh * kw
        scale = 1.0 / jnp.sqrt(jnp.asarray((n * ho * wo) * fan_in_per, dtype=accum.dtype))
        return accum * scale

    # grouped case: contiguous partition
    assert Cin % groups == 0 and Cout % groups == 0, "Cin and Cout must be divisible by groups"
    Cin_per = Cin // groups
    Cout_per = Cout // groups

    patches_g = patches.reshape(n, ho, wo, kh, kw, groups, Cin_per)  # (N,Ho,Wo,Kh,Kw,G,Cin_per)
    out_g = out_grad.reshape(n, ho, wo, groups, Cout_per)  # (N,Ho,Wo,G,Cout_per)

    accum_g = jnp.einsum(
        "n h w k l g c, n h w g o -> k l g c o", patches_g, out_g
    )  # (Kh,Kw,G,Cin_per,Cout_per)
    fan_in_per = Cin_per * kh * kw
    scale = 1.0 / jnp.sqrt(jnp.asarray((n * ho * wo) * fan_in_per, dtype=accum_g.dtype))
    accum_g = accum_g * scale

    accum = accum_g.transpose(0, 1, 3, 2, 4).reshape(kh, kw, Cin, Cout)
    return accum


def conv_backward_with_threshold(
    x: Array,
    out_grad: Array,
    out_grad_hat: Array,
    threshold: Array,
    kernel_shape: tuple[int, int],
    groups: int = 1,
    strides: IntPair = (1, 1),
    padding_mode: str | Callable[..., str] | None = None,
    lhs_dilation: IntPair = (1, 1),
    rhs_dilation: IntPair = (1, 1),
) -> Array:
    """
    Kernel-gradient-like accumulator with elementwise threshold gating.

    Only uses out_grad elements where (out_grad * out_grad_hat) < threshold.
    Groupwise and geometry semantics match conv_backward. The returned accumulator
    uses the same √-rescaling:

        1 / sqrt((N * Ho * Wo) * fan_in_per).

    Parameters
    ----------
    x : Array
        Input tensor (N, H, W, Cin).
    out_grad : Array
        Signal at conv output (N, Ho, Wo, Cout).
    out_grad_hat : Array
        Secondary signal used for gating (N, Ho, Wo, Cout).
    threshold : scalar
        Gating threshold. Same sign semantics as original code:
        keep positions where (out_grad * out_grad_hat < threshold).
    kernel_shape : (Kh, Kw)
        Spatial kernel size.
    groups : int
        Number of contiguous groups.
    strides, padding_mode, lhs_dilation, rhs_dilation :
        Patch extraction geometry.

    Returns
    -------
    Array
        Gated, variance-normalized kernel accumulator (Kh, Kw, Cin, Cout).

    """
    n, ho, wo, Cout = out_grad.shape
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
    patches = patches.reshape(patches.shape[0], patches.shape[1], patches.shape[2], kh, kw, Cin)

    mask = (out_grad * out_grad_hat < threshold).astype(out_grad.dtype)
    out_masked = out_grad * mask

    if groups == 1:
        accum = jnp.einsum("n h w k l c, n h w o -> k l c o", patches, out_masked)
        fan_in_per = Cin * kh * kw
        scale = 1.0 / jnp.sqrt(jnp.asarray((n * ho * wo) * fan_in_per, dtype=accum.dtype))
        return accum * scale

    assert Cin % groups == 0 and Cout % groups == 0, "Cin and Cout must be divisible by groups"
    Cin_per = Cin // groups
    Cout_per = Cout // groups

    patches_g = patches.reshape(n, ho, wo, kh, kw, groups, Cin_per)
    out_g = out_masked.reshape(n, ho, wo, groups, Cout_per)

    accum_g = jnp.einsum("n h w k l g c, n h w g o -> k l g c o", patches_g, out_g)
    fan_in_per = Cin_per * kh * kw
    scale = 1.0 / jnp.sqrt(jnp.asarray((n * ho * wo) * fan_in_per, dtype=accum_g.dtype))
    accum_g = accum_g * scale

    accum = accum_g.transpose(0, 1, 3, 2, 4).reshape(kh, kw, Cin, Cout)
    return accum


def conv_transpose_backward_with_threshold(
    x: Array,
    out_grad: Array,
    out_grad_hat: Array,
    threshold: Array,
    kernel_shape: tuple[int, int],
    stride: IntPair,
    groups: int = 1,
    padding_mode: str | Callable[..., str] | None = None,
) -> Array:
    """
    Kernel accumulator for transposed-convolution geometry with gating and rescaling.

    Matches the geometry of conv_transpose_forward (rhs_dilation=stride).
    Uses the same gating rule and √-rescaling as the other backward helpers.

    Parameters
    ----------
    x : Array
        Input tensor (N, H, W, Cin).
    out_grad : Array
        Signal at transposed conv output (N, Ho, Wo, Cout).
    out_grad_hat : Array
        Secondary signal used for gating.
    threshold : scalar
        Gating threshold.
    kernel_shape : (Kh, Kw)
        Spatial kernel shape.
    stride : int or (sh, sw)
        Upsampling factor used in forward transposed conv.
    groups : int
        Number of contiguous groups.
    padding_mode : str/callable/None
        Padding mode for pad_2d.

    Returns
    -------
    Array
        Kernel accumulator shaped (Kh, Kw, Cin, Cout), variance-normalized.

    """
    n, ho, wo, Cout = out_grad.shape
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
    patches = patches.reshape(patches.shape[0], patches.shape[1], patches.shape[2], kh, kw, Cin)

    mask = (out_grad * out_grad_hat < threshold).astype(patches.dtype)
    out_masked = out_grad * mask

    if groups == 1:
        accum = jnp.einsum("n h w k l c, n h w o -> k l c o", patches, out_masked)
        fan_in_per = Cin * kh * kw
        scale = 1.0 / jnp.sqrt(jnp.asarray((n * ho * wo) * fan_in_per, dtype=accum.dtype))
        return accum * scale

    assert Cin % groups == 0 and Cout % groups == 0, "Cin and Cout must be divisible by groups"
    Cin_per = Cin // groups
    Cout_per = Cout // groups

    patches_g = patches.reshape(n, ho, wo, kh, kw, groups, Cin_per)
    out_g = out_masked.reshape(n, ho, wo, groups, Cout_per)

    accum_g = jnp.einsum("n h w k l g c, n h w g o -> k l g c o", patches_g, out_g)
    fan_in_per = Cin_per * kh * kw
    scale = 1.0 / jnp.sqrt(jnp.asarray((n * ho * wo) * fan_in_per, dtype=accum_g.dtype))
    accum_g = accum_g * scale

    accum = accum_g.transpose(0, 1, 3, 2, 4).reshape(kh, kw, Cin, Cout)
    return accum
