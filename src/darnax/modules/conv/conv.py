import operator
from typing import Self
from collections.abc import Callable
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from darnax.modules.interfaces import Adapter, Layer
from jax import Array
from jax.typing import DTypeLike
from jax.tree_util import tree_reduce
from darnax.utils.typing import PyTree
from darnax.modules.conv.utils import (
    fetch_tuple_from_arg,
    conv_backward_with_threshold,
    conv_transpose_backward_with_threshold,
    conv_transpose_forward,
    conv_forward,
    pad_2d,
)


KeyArray = Array


class Conv2D(Adapter):
    """Implement convolutional Win."""

    kernel: Array
    threshold: Array
    strength: float = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    kernel_size: tuple[int, int] = eqx.field(static=True)
    padding_mode: str | Callable[..., str] | None = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        threshold: float,
        strength: float,
        key: KeyArray,
        stride: int | tuple[int, int] = 1,
        padding_mode: str | Callable[..., str] | None = None,
        dtype: DTypeLike = jnp.float32,
    ):
        """Save properties and init kernel."""
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = fetch_tuple_from_arg(stride)
        self.kernel_size = fetch_tuple_from_arg(kernel_size)
        self.padding_mode = padding_mode
        self.threshold = jnp.asarray(threshold, dtype=dtype)  # scalar for simplicity
        self.strength = strength

        key, init_key = jax.random.split(key)
        kh, kw = self.kernel_size
        self.kernel = (
            jax.random.normal(
                key=init_key, shape=(kh, kw, self.in_channels, self.out_channels), dtype=dtype
            )
            / (kh * kw * self.in_channels * self.out_channels) ** 0.5
        )

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Implement simple convolution. With optional padding."""
        # conv_forward already handles padding via pad_2d and uses dimension_numbers explicitly
        return self.strength * conv_forward(x, self.kernel, self.stride, self.padding_mode)

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Update parameters using conv backward with threshold/margin."""
        dW = conv_backward_with_threshold(
            x, y, y_hat, self.threshold, self.kernel_size, self.stride, self.padding_mode
        )
        # zero-update tree with same structure (use tree_map from jax)
        zero_update = jax.tree_util.tree_map(jnp.zeros_like, self, is_leaf=eqx.is_inexact_array)
        new_self: Self = eqx.tree_at(lambda m: m.kernel, zero_update, dW)
        return new_self


class Conv2DRecurrentDiscrete(Layer):
    """Recurrent convolutional layer with constant spatial + channel size.

    - NHWC throughout.
    - Cin = Cout = channels.
    - Grouped convolution via `feature_group_count=groups`.
    - Enforces a constant "central_element" self-connection set to j_d
      at forward-time (stored kernel keeps those entries zero to avoid drift).
    """

    kernel: Array
    threshold: Array
    j_d: Array

    channels: int = eqx.field(static=True)
    groups: int = eqx.field(static=True)
    kernel_size: tuple[int, int] = eqx.field(static=True)
    padding_mode: str | Callable[..., str] | None = eqx.field(static=True)
    central_element: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        kernel_size: int | tuple[int, int],
        groups: int,
        j_d: float,
        threshold: float,
        key: KeyArray,
        padding_mode: str | Callable[..., str] = "constant",
        dtype: DTypeLike = jnp.float32,
    ):
        self.channels = int(channels)
        self.groups = int(groups)
        self.kernel_size = fetch_tuple_from_arg(kernel_size)
        self.padding_mode = padding_mode
        self.threshold = jnp.asarray(threshold, dtype=dtype)
        self.j_d = jnp.asarray(j_d, dtype=dtype)

        kh, kw = self.kernel_size

        # Spatial-size preservation with our symmetric pad_2d in conv_forward requires odd kernels.
        if (kh % 2) == 0 or (kw % 2) == 0:
            raise ValueError(
                f"Conv2DRecurrentDiscrete requires odd kernel_size for exact same H,W "
                f"with symmetric padding. Got kernel_size={self.kernel_size}."
            )

        # Group constraints: Cin=Cout=channels, so groups must divide channels
        if self.channels % self.groups != 0:
            raise ValueError(
                f"`groups` must divide `channels`. Got channels={self.channels}, groups={self.groups}."
            )

        self.central_element = (kh // 2, kw // 2)

        cin_g = self.channels // self.groups
        cout = self.channels

        key, init_key = jax.random.split(key)
        self.kernel = (
            jax.random.normal(init_key, shape=(kh, kw, cin_g, cout), dtype=dtype)
            / (kh * kw * cin_g * cout) ** 0.5
        )
        # Write j_d into the constrained diagonal entries and then project them out in the stored kernel.
        # We intentionally keep stored kernel constrained entries zero (projected) so they don't drift during learning.
        self.kernel = self._set_jd_constraint(self.kernel)
        self.kernel = self._project_kernel(self.kernel)

    def _central_diag_mask(self) -> Array:
        """Mask (cin_g, cout) with ones exactly at the self-connection positions."""
        cin_g = self.channels // self.groups
        cout_g = self.channels // self.groups
        cout = self.channels

        c = jnp.arange(cout)
        local = c % cout_g  # (cout,)
        mask = jnp.zeros((cin_g, cout), dtype=self.kernel.dtype)
        mask = mask.at[local, c].set(1)
        return mask

    def _set_jd_constraint(self, k: Array) -> Array:
        """Write j_d into the constrained diagonal entries at central_element (non-destructive)."""
        ch, cw = self.central_element
        mask = self._central_diag_mask()  # (cin_g, cout)
        center = k[ch, cw, :, :]
        center = center * (1 - mask) + self.j_d * mask
        k = k.at[ch, cw, :, :].set(center)
        return k

    def _project_kernel(self, k: Array) -> Array:
        """Project parameters so the constrained entries are zero in the stored kernel."""
        ch, cw = self.central_element
        mask = self._central_diag_mask()
        center = k[ch, cw, :, :] * (1 - mask)  # force constrained entries to 0
        k = k.at[ch, cw, :, :].set(center)
        return k

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Perform grouped convolution using an effective kernel where the central diag entries equal j_d."""
        kh, kw = self.kernel_size

        # create effective kernel with j_d in the constrained central positions
        kernel_effective = self._set_jd_constraint(self.kernel)

        # symmetric half-kernel padding so odd kernels preserve H,W for stride=1
        pad_h = kh // 2
        pad_w = kw // 2
        x_pad = pad_2d(x, pad_h, pad_w, self.padding_mode)

        y: Array = lax.conv_general_dilated(
            lhs=x_pad,
            rhs=kernel_effective,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.groups,
        )
        return y

    def activation(self, x: Array) -> Array:
        """Binarize activation."""
        return jnp.sign(x)

    def reduce(self, h: PyTree) -> Array:
        return jnp.asarray(tree_reduce(operator.add, h))

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Compute group-aware correlation updates and zero the constrained center updates."""
        kh, kw = self.kernel_size
        cin = self.channels
        cout = self.channels
        g = self.groups
        cin_g = cin // g
        cout_g = cout // g

        dW_full = conv_backward_with_threshold(
            x=x,
            y=y,
            y_hat=y_hat,
            threshold=self.threshold,
            kernel_shape=(kh, kw),
            strides=(1, 1),
            padding_mode=self.padding_mode,
        )

        # Reshape to (Kh, Kw, G, Cin_g, G, Cout_g)
        dW_full = dW_full.reshape(kh, kw, g, cin_g, g, cout_g)

        # Keep only within-group connections (diagonal over the two group axes)
        dW_diag = jnp.diagonal(dW_full, axis1=2, axis2=4)

        # Move G next to Cout_g and flatten back to Cout:
        dW_grouped = jnp.transpose(dW_diag, (0, 1, 2, 4, 3)).reshape(kh, kw, cin_g, cout)

        # Enforce "central_element" is not learnable: kill updates on the diagonal positions
        ch, cw = self.central_element
        mask = self._central_diag_mask()
        center = dW_grouped[ch, cw, :, :] * (1 - mask)
        dW_grouped = dW_grouped.at[ch, cw, :, :].set(center)

        zero_update = jax.tree_util.tree_map(jnp.zeros_like, self, is_leaf=eqx.is_inexact_array)
        new_self: Self = eqx.tree_at(lambda m: m.kernel, zero_update, dW_grouped)
        return new_self


class Conv2DTranspose(Adapter):
    """Transposed convolution (deconvolution) analog of Conv2D."""

    kernel: Array
    threshold: Array
    strength: float = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)  # stored as scalar
    kernel_size: tuple[int, int] = eqx.field(static=True)
    padding_mode: str | Callable[..., str] | None = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        threshold: float,
        strength: float,
        key: KeyArray,
        stride: int | tuple[int, int] = 1,
        padding_mode: str | Callable[..., str] | None = None,
        dtype: DTypeLike = jnp.float32,
    ):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        stride_tuple = fetch_tuple_from_arg(stride)
        # store a single integer for convenience; conv_transpose_forward uses this scalar for both dims
        self.stride = int(stride_tuple[0])
        self.kernel_size = fetch_tuple_from_arg(kernel_size)
        self.padding_mode = padding_mode
        self.threshold = jnp.asarray(threshold, dtype=dtype)
        self.strength = float(strength)

        key, init_key = jax.random.split(key)
        kh, kw = self.kernel_size
        self.kernel = (
            jax.random.normal(
                key=init_key,
                shape=(kh, kw, self.in_channels, self.out_channels),
                dtype=dtype,
            )
            / (kh * kw * self.in_channels * self.out_channels) ** 0.5
        )

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        # conv_transpose_forward handles padding / dilation semantics
        return conv_transpose_forward(x, self.kernel, self.stride, self.padding_mode)

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        dW = conv_transpose_backward_with_threshold(
            x=x,
            y=y,
            y_hat=y_hat,
            threshold=self.threshold,
            kernel_shape=self.kernel_size,
            stride=self.stride,
            padding_mode=self.padding_mode,
        )
        zero_update = jax.tree_util.tree_map(jnp.zeros_like, self, is_leaf=eqx.is_inexact_array)
        return eqx.tree_at(lambda m: m.kernel, zero_update, dW)
