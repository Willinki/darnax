from typing import TYPE_CHECKING
from collections.abc import Callable
import jax.numpy as jnp
from jax import lax
from jax import Array


if TYPE_CHECKING:
    KeyArray = Array


def fetch_tuple_from_arg(arg: int | tuple[int, int]):
    """Return fixed shape of arg and assert type."""
    assert isinstance(arg, (int, tuple))
    if isinstance(arg, tuple):
        assert len(arg) == 2
        assert all(isinstance(x, int) for x in arg)
        arg_tuple = arg
    else:
        arg_tuple = (arg, arg)
    return arg_tuple


def pad_2d(x: Array, pad_h: int, pad_w: int, mode: str | Callable[..., str] | None):
    """Apply various types of padding to an image.

    Its supposing an image of shape [batch, height, width, channels].

    For proper modes see: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.pad.html.
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


def conv_backward(
    x: Array,  # (N, H, W, Cin)  NHWC
    y: Array,  # (N, Ho, Wo, Cout) NHWC
    kernel_shape: tuple[int, int],  # (Kh, Kw)
    strides: tuple[int, int] = (1, 1),
    padding_mode: str | Callable[..., str] | None = None,
    lhs_dilation=(1, 1),
    rhs_dilation=(1, 1),
) -> Array:
    """Accumulate conv-correlations across strides and sum.

    No margin based update.
    """
    # shapes / normalization
    n, ho, wo, _ = y.shape
    kh, kw = kernel_shape
    Cin = x.shape[-1]
    strides = fetch_tuple_from_arg(strides)

    # pad x in the same symmetric way as conv_forward (so odd kernels preserve H,W)
    pad_h = kh // 2
    pad_w = kw // 2
    x_pad = pad_2d(x, pad_h, pad_w, padding_mode)

    # extract patches. We use VALID since we already padded x_pad.
    patches = lax.conv_general_dilated_patches(
        x_pad,
        filter_shape=(kh, kw),
        window_strides=strides,
        padding="VALID",
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    # raw shape: (N, H_p, W_p, Kh*Kw*Cin)
    H_p, W_p = patches.shape[1], patches.shape[2]

    # reshape to per-patch layout
    patches = patches.reshape(patches.shape[0], H_p, W_p, kh, kw, Cin)

    # Correlate: sum over N,Ho,Wo -> (Kh, Kw, Cin, Cout)
    corr = jnp.einsum("nhwuvc,nhwo->uvco", patches, y)
    corr = corr / (n * ho * wo) ** 0.5
    return corr


def conv_backward_with_threshold(
    x: Array,  # (N, H, W, Cin)  NHWC
    y: Array,  # (N, Ho, Wo, Cout) NHWC
    y_hat: Array,  # (N, Ho, Wo, Cout) NHWC
    threshold: Array,  # scalar
    kernel_shape: tuple[int, int],  # (Kh, Kw)
    strides: tuple[int, int] = (1, 1),
    padding_mode: str | Callable[..., str] | None = None,
    lhs_dilation=(1, 1),
    rhs_dilation=(1, 1),
) -> Array:
    """Accumulate conv-correlations across strides and sum. With margin."""
    n, ho, wo, _ = y.shape
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
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    H_p, W_p = patches.shape[1], patches.shape[2]
    patches = patches.reshape(patches.shape[0], H_p, W_p, kh, kw, Cin)

    threshold_mask = (y * y_hat < threshold).astype(patches.dtype)
    corr = jnp.einsum("nhwuvc,nhwo->uvco", patches, y * threshold_mask)
    corr = corr / (n * ho * wo) ** 0.5
    return corr


def conv_forward(
    x: Array,
    kernel: Array,
    stride: int | tuple[int, int] = 1,
    padding_mode: str | Callable[..., str] | None = None,
) -> Array:
    """Do a simple padding plus convolution (NHWC / HWIO)."""
    # normalize stride to a (sh, sw) tuple
    stride = fetch_tuple_from_arg(stride)

    # explicit layouts: NHWC input, HWIO kernel, NHWC output
    x_pad = pad_2d(x, kernel.shape[0] // 2, kernel.shape[1] // 2, padding_mode)
    y: Array = lax.conv_general_dilated(
        lhs=x_pad,
        rhs=kernel,
        window_strides=stride,
        padding="VALID",  # we already applied padding in x_pad
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y


def conv_transpose_forward(
    x: Array,
    kernel: Array,
    stride: int | tuple[int, int] = 1,
    padding_mode: str | Callable[..., str] | None = None,
) -> Array:
    """Forward transposed convolution implemented via lax.conv_general_dilated.

    Uses rhs_dilation=stride to emulate conv-transpose while preserving dimension order.
    """
    stride = fetch_tuple_from_arg(stride)
    kh, kw, c_in_k, c_out = kernel.shape
    assert c_in_k == x.shape[-1], "kernel in-channels mismatch"

    x_pad = pad_2d(x, kh // 2, kw // 2, padding_mode)

    y: Array = lax.conv_general_dilated(
        lhs=x_pad,
        rhs=kernel,
        window_strides=(1, 1),  # dilation on rhs creates the "transpose" upsample effect
        padding=((kh - 1, kh - 1), (kw - 1, kw - 1)),  # chosen to grow output with stride
        lhs_dilation=(1, 1),
        rhs_dilation=stride,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y


def conv_transpose_backward_with_threshold(
    x: Array,  # (N, H, W, Cin)  NHWC
    y: Array,  # (N, Ho, Wo, Cout) NHWC
    y_hat: Array,  # (N, Ho, Wo, Cout) NHWC
    threshold: Array,  # scalar
    kernel_shape: tuple[int, int],  # (Kh, Kw)
    stride: int,
    padding_mode: str | Callable[..., str] | None = None,
) -> Array:
    """Correlation-style update for conv_transpose_forward implemented via rhs_dilation.

    This mirrors conv_transpose_forward exactly:
      - pad_2d(x, kh//2, kw//2)
      - patches extracted with rhs_dilation=(stride,stride)
      - padding=((kh-1,kh-1),(kw-1,kw-1))
    """
    n, ho, wo, _ = y.shape
    kh, kw = kernel_shape
    cin = x.shape[-1]

    # Must match conv_transpose_forward input padding
    x_pad = pad_2d(x, kh // 2, kw // 2, padding_mode)

    # Must match conv_transpose_forward convolution geometry
    patches = lax.conv_general_dilated_patches(
        x_pad,
        filter_shape=(kh, kw),
        window_strides=(1, 1),
        padding=((kh - 1, kh - 1), (kw - 1, kw - 1)),
        lhs_dilation=(1, 1),
        rhs_dilation=(int(stride), int(stride)),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    # patches: (N, Ho, Wo, Kh*Kw*Cin)
    patches = patches.reshape(*patches.shape[:-1], kh, kw, cin)  # (N,Ho,Wo,Kh,Kw,Cin)

    thr = (y * y_hat < threshold).astype(patches.dtype)
    corr = jnp.einsum("nhwuvc,nhwo->uvco", patches, y * thr)

    corr = corr / (n * ho * wo) ** 0.5
    return corr
