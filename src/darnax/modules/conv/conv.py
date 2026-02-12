"""Convolutional neural network modules with Hebbian-style learning.

This module implements convolution layers with activity-dependent plasticity
rules instead of traditional backpropagation. Weight updates are computed
using thresholded correlations between pre- and post-synaptic signals.

Classes
-------
Conv2D : Feedforward convolution adapter
Conv2DTranspose : Transposed convolution adapter
Conv2DRecurrentDiscrete : Recurrent grouped convolution with fixed diagonal

Variance Normalization
----------------------
All modules maintain unit variance through the network:
    - Forward: Kernel init with Var(W) = 1/fan_in
    - Backward: Gradient normalized by 1/sqrt(N × Ho × Wo) ONLY
    - Weight decay: Normalized identically to gradients

CRITICAL: The backward pass normalization does NOT include fan_in.
The gradient accumulates over spatial/batch dimensions (N × Ho × Wo),
not over kernel elements (fan_in).

Examples
--------
>>> import jax
>>> from jax import random
>>>
>>> # Create a simple conv adapter
>>> key = random.PRNGKey(0)
>>> conv = Conv2D(
...     in_channels=64,
...     out_channels=128,
...     kernel_size=3,
...     threshold=0.0,
...     strength=1.0,
...     key=key,
...     lr=0.01,
...     weight_decay=0.0001
... )
>>>
>>> # Forward pass
>>> x = random.normal(key, (4, 32, 32, 64))
>>> y = conv(x)
>>>
>>> # Backward pass (Hebbian update)
>>> y_hat = random.normal(key, y.shape)
>>> update = conv.backward(x, y, y_hat)

"""

import operator
from collections.abc import Callable
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.tree_util import tree_reduce
from jax.typing import DTypeLike

from darnax.modules.conv.utils import (
    conv_backward_with_threshold,
    conv_forward,
    conv_transpose_backward_with_threshold,
    conv_transpose_forward,
    fetch_tuple_from_arg,
    pad_2d,
)
from darnax.modules.interfaces import Adapter, Layer
from darnax.utils.typing import PyTree

KeyArray = Array


class Conv2D(Adapter):
    """Feedforward 2D convolution with Hebbian-style learning.

    This adapter implements a standard 2D convolution with activity-dependent
    weight updates. Unlike backpropagation, updates are computed using local
    correlation signals gated by a threshold.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple[int, int]
        Size of the convolution kernel. If int, same size for both dimensions.
    threshold : float
        Threshold for gating updates. Positions where
        (y * y_hat < threshold) receive updates.
    strength : float
        Multiplicative factor applied to output. Controls the effective
        learning rate in the forward pass.
    key : KeyArray
        JAX random key for weight initialization.
    stride : int or tuple[int, int], default=1
        Stride for the convolution.
    padding_mode : str, callable, or None, default=None
        Padding mode ('constant', 'edge', 'reflect', or None).
    dtype : DTypeLike, default=jnp.float32
        Data type for parameters.
    lr : float, default=1.0
        Learning rate multiplier for gradient updates.
    weight_decay : float, default=0.0
        Weight decay coefficient (L2 regularization).

    Attributes
    ----------
    kernel : Array
        Convolution kernel of shape (Kh, Kw, in_channels, out_channels).
    threshold : Array
        Scalar threshold value.
    lr : Array
        Scalar learning rate.
    weight_decay : Array
        Scalar weight decay coefficient.
    strength : float
        Output scaling factor (static).
    in_channels : int
        Number of input channels (static).
    out_channels : int
        Number of output channels (static).
    stride : tuple[int, int]
        Convolution stride (static).
    kernel_size : tuple[int, int]
        Kernel spatial size (static).
    padding_mode : str, callable, or None
        Padding mode (static).

    Notes
    -----
    Weight Initialization:
        Kernels are initialized from N(0, 1/fan_in) where fan_in = Kh × Kw × Cin.
        This ensures unit output variance for unit input variance.

    Variance Normalization:
        - Forward: Output variance = strength² × input variance
        - Backward: Gradient normalized by 1/sqrt(N × Ho × Wo) ONLY
        - Weight decay: Scaled identically to gradients

    CRITICAL: The backward normalization does NOT include fan_in, unlike what
    was previously documented. The gradient sums over (N, Ho, Wo) spatial/batch
    dimensions, so we normalize by sqrt(N × Ho × Wo) only.

    Examples
    --------
    >>> key = jax.random.PRNGKey(0)
    >>> conv = Conv2D(
    ...     in_channels=3,
    ...     out_channels=16,
    ...     kernel_size=5,
    ...     threshold=0.0,
    ...     strength=1.0,
    ...     key=key
    ... )
    >>> x = jax.random.normal(key, (1, 28, 28, 3))
    >>> y = conv(x)
    >>> y.shape
    (1, 28, 28, 16)

    """

    kernel: Array
    threshold: Array
    lr: Array
    weight_decay: Array

    strength: float = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    stride: tuple[int, int] = eqx.field(static=True)
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
        *,
        lr: float = 1.0,
        weight_decay: float = 0.0,
    ):
        """Initialize Conv2D adapter with proper variance scaling."""
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = fetch_tuple_from_arg(stride)
        self.kernel_size = fetch_tuple_from_arg(kernel_size)
        self.padding_mode = padding_mode
        self.threshold = jnp.asarray(threshold, dtype=dtype)
        self.strength = float(strength)

        kh, kw = self.kernel_size
        fan_in = kh * kw * self.in_channels

        self.lr = jnp.asarray(lr, dtype=dtype)
        # Store weight decay at base scale (no fan_in pre-normalization)
        self.weight_decay = jnp.asarray(weight_decay, dtype=dtype)

        # Initialize kernel: Var(W) = 1/fan_in for unit output variance
        key, init_key = jax.random.split(key)
        self.kernel = jax.random.normal(
            key=init_key,
            shape=(kh, kw, self.in_channels, self.out_channels),
            dtype=dtype,
        ) / jnp.sqrt(jnp.asarray(fan_in, dtype=dtype))

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Forward pass through the convolution.

        Parameters
        ----------
        x : Array
            Input tensor of shape (N, H, W, Cin).
        rng : KeyArray or None, default=None
            Random key (unused, included for interface compatibility).

        Returns
        -------
        Array
            Output tensor of shape (N, Ho, Wo, Cout).

        Notes
        -----
        Output variance = strength² × input variance (for unit variance input).

        """
        return self.strength * conv_forward(x, self.kernel, self.stride, self.padding_mode)

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Compute Hebbian-style weight update.

        Parameters
        ----------
        x : Array
            Input tensor of shape (N, H, W, Cin).
        y : Array
            Primary output signal of shape (N, Ho, Wo, Cout).
        y_hat : Array
            Secondary signal for gating of shape (N, Ho, Wo, Cout).
        gate : Array or None, default=None
            Optional global gate (unused, for interface compatibility).

        Returns
        -------
        Self
            Update tree with same structure as self. All leaves are zero except
            the kernel, which contains the computed update.

        Notes
        -----
        Update Rule:
            dW = lr × gradient + weight_decay × decay_scale × W

            where gradient is computed only at positions where (y * y_hat < threshold).

        Normalization:
            Both gradient and weight decay are normalized by 1/sqrt(N × Ho × Wo)
            to maintain consistent magnitude. This ensures unit variance gradients
            when inputs have unit variance.

        Examples
        --------
        >>> conv = Conv2D(64, 128, 3, threshold=0.0, strength=1.0, key=key)
        >>> x = jax.random.normal(key, (4, 32, 32, 64))
        >>> y = conv(x)
        >>> y_hat = jax.random.normal(key, y.shape)
        >>> update = conv.backward(x, y, y_hat)
        >>> # update.kernel contains the weight update

        """
        dW = conv_backward_with_threshold(
            x=x,
            y=y,
            y_hat=y_hat,
            threshold=self.threshold,
            kernel_shape=self.kernel_size,
            groups=1,
            strides=self.stride,
            padding_mode=self.padding_mode,
        )

        # Weight decay with same normalization as gradient: 1/sqrt(N × Ho × Wo)
        n, ho, wo, _ = y.shape
        decay_scale = 1.0 / jnp.sqrt(jnp.asarray(n * ho * wo, dtype=dW.dtype))

        dW = self.lr * dW + self.weight_decay * decay_scale * self.kernel

        zero_update: Self = jax.tree_util.tree_map(
            jnp.zeros_like, self, is_leaf=eqx.is_inexact_array
        )
        update: Self = eqx.tree_at(lambda m: m.kernel, zero_update, dW)
        return update


class Conv2DRecurrentDiscrete(Layer):
    """Recurrent grouped 2D convolution with fixed diagonal self-connections.

    This layer implements a recurrent convolution where input and output have
    the same number of channels (Cin = Cout = channels). The central kernel
    element has diagonal entries fixed to a constant value (j_d) that never
    changes during training. This creates stable self-connections in the
    recurrent dynamics.

    Parameters
    ----------
    channels : int
        Number of channels (same for input and output).
    kernel_size : int or tuple[int, int]
        Size of convolution kernel. Must be odd for symmetric padding.
    groups : int
        Number of groups for grouped convolution. Must divide channels evenly.
        Channels are partitioned contiguously into groups.
    j_d : float
        Fixed value for diagonal self-connections at the central kernel element.
        These connections are never updated during training.
    threshold : float
        Threshold for gating weight updates.
    key : KeyArray
        JAX random key for initialization.
    padding_mode : str or callable, default='constant'
        Padding mode for convolution.
    dtype : DTypeLike, default=jnp.float32
        Data type for parameters.
    lr : float
        Learning rate multiplier.
    weight_decay : float
        Weight decay coefficient.

    Attributes
    ----------
    kernel : Array
        Convolution kernel of shape (Kh, Kw, channels//groups, channels).
    threshold : Array
        Scalar threshold value.
    j_d : Array
        Fixed diagonal value.
    lr : Array
        Scalar learning rate.
    weight_decay : Array
        Scalar weight decay coefficient.
    update_mask : Array
        Binary mask (same shape as kernel) where constrained parameters
        (diagonal self-connections) are 0 and trainable parameters are 1.
    channels : int
        Number of channels (static).
    groups : int
        Number of groups (static).
    kernel_size : tuple[int, int]
        Kernel spatial size (static).
    padding_mode : str, callable, or None
        Padding mode (static).
    central_element : tuple[int, int]
        Index of central kernel element (static).

    Raises
    ------
    ValueError
        If kernel_size is even (must be odd for symmetric padding).
        If channels is not divisible by groups.

    Notes
    -----
    Grouping:
        Channels are partitioned contiguously. For channels=64 and groups=4:
        - Group 0: channels 0-15
        - Group 1: channels 16-31
        - Group 2: channels 32-47
        - Group 3: channels 48-63

    Diagonal Constraint:
        At the central kernel position (kh//2, kw//2), the diagonal entries
        connecting each channel to itself are fixed to j_d. The update mask
        zeros out gradients for these parameters.

    Variance Normalization:
        - Backward: Gradient normalized by 1/sqrt(N × Ho × Wo) ONLY
        - Weight decay: Scaled identically to gradients

    Examples
    --------
    >>> key = jax.random.PRNGKey(0)
    >>> layer = Conv2DRecurrentDiscrete(
    ...     channels=64,
    ...     kernel_size=3,
    ...     groups=4,
    ...     j_d=1.0,
    ...     threshold=0.0,
    ...     key=key,
    ...     lr=0.01,
    ...     weight_decay=0.0001
    ... )
    >>> x = jax.random.normal(key, (1, 32, 32, 64))
    >>> y = layer(x)
    >>> y.shape
    (1, 32, 32, 64)

    """

    kernel: Array
    threshold: Array
    j_d: Array
    lr: Array
    weight_decay: Array
    update_mask: Array  # same shape as kernel; 0 blocks constrained params

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
        *,
        lr: float,
        weight_decay: float,
    ):
        """Initialize recurrent convolution with diagonal constraint."""
        self.channels = int(channels)
        self.groups = int(groups)
        self.kernel_size = fetch_tuple_from_arg(kernel_size)
        self.padding_mode = padding_mode

        kh, kw = self.kernel_size
        if (kh % 2) == 0 or (kw % 2) == 0:
            raise ValueError(
                f"kernel_size must be odd for same-size padding. Got {self.kernel_size}."
            )

        if self.channels % self.groups != 0:
            raise ValueError(
                f"`groups` must divide `channels`. Got channels={self.channels}, groups={self.groups}."
            )

        self.central_element = (kh // 2, kw // 2)

        self.threshold = jnp.asarray(threshold, dtype=dtype)
        self.j_d = jnp.asarray(j_d, dtype=dtype)
        self.lr = jnp.asarray(lr, dtype=dtype)

        cin_g = self.channels // self.groups  # per-group input channels for JAX grouped conv
        cout = self.channels  # enforce Cin=Cout=channels

        # fan-in per output activation (grouped)
        fan_in = kh * kw * cin_g
        # Store weight decay at base scale (no fan_in pre-normalization)
        self.weight_decay = jnp.asarray(weight_decay, dtype=dtype)

        key, init_key = jax.random.split(key)
        self.kernel = jax.random.normal(
            init_key, shape=(kh, kw, cin_g, cout), dtype=dtype
        ) / jnp.sqrt(jnp.asarray(fan_in, dtype=dtype))

        # Build update mask once (0 at constrained params, 1 elsewhere), then hard-set j_d at init.
        self.update_mask = self._build_update_mask(dtype=dtype)
        self.kernel = self._apply_jd_constraint(self.kernel)

    # ---------- constraint helpers ----------
    def _central_diag_mask(self) -> Array:
        """Create mask for channel-diagonal self-connections.

        Returns
        -------
        Array
            Binary mask of shape (cin_g, cout) with 1s at diagonal positions
            where each output channel connects to its corresponding input
            channel within the same group.

        Notes
        -----
        For channels=64, groups=4 (cin_g=16, cout_g=16):
        - Output channel 0 connects to input 0 (within group 0)
        - Output channel 16 connects to input 0 (within group 1)
        - Output channel 17 connects to input 1 (within group 1)
        - etc.

        """
        cin_g = self.channels // self.groups
        cout = self.channels
        cout_g = self.channels // self.groups

        c = jnp.arange(cout)  # [0..cout-1]
        local = c % cout_g  # maps output channel -> local index within its group
        mask = jnp.zeros((cin_g, cout), dtype=self.kernel.dtype)
        return mask.at[local, c].set(1)

    def _build_update_mask(self, dtype: DTypeLike) -> Array:
        """Build mask to block updates to constrained parameters.

        Returns
        -------
        Array
            Binary mask of shape (Kh, Kw, cin_g, cout) where:
            - 0 at constrained diagonal entries (central element)
            - 1 everywhere else (trainable parameters)

        Notes
        -----
        This mask is applied to gradients to ensure diagonal self-connections
        at the central kernel element remain fixed at j_d.

        """
        kh, kw = self.kernel_size
        cin_g = self.channels // self.groups
        cout = self.channels
        ch, cw = self.central_element

        mask = jnp.ones((kh, kw, cin_g, cout), dtype=dtype)
        diag = self._central_diag_mask().astype(dtype)  # 1 where constrained
        return mask.at[ch, cw, :, :].set(1.0 - diag)  # 0 at constrained center entries

    def _apply_jd_constraint(self, k: Array) -> Array:
        """Write j_d into constrained diagonal entries.

        Parameters
        ----------
        k : Array
            Kernel array of shape (Kh, Kw, cin_g, cout).

        Returns
        -------
        Array
            Kernel with diagonal self-connections at central element set to j_d.

        Notes
        -----
        Called once at initialization to set the fixed diagonal values.

        """
        ch, cw = self.central_element
        diag = self._central_diag_mask().astype(k.dtype)
        center = k[ch, cw, :, :]
        center = center * (1.0 - diag) + self.j_d * diag
        return k.at[ch, cw, :, :].set(center)

    # ---------- forward ----------
    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Forward pass through grouped convolution.

        Parameters
        ----------
        x : Array
            Input tensor of shape (N, H, W, channels).
        rng : KeyArray or None, default=None
            Random key (unused, for interface compatibility).

        Returns
        -------
        Array
            Output tensor of shape (N, H, W, channels).

        Raises
        ------
        ValueError
            If x.shape[-1] != self.channels.

        Notes
        -----
        Uses JAX's feature_group_count for grouped convolution.
        Output size matches input size (same padding).

        """
        if x.shape[-1] != self.channels:
            raise ValueError(f"Expected x.shape[-1]==channels=={self.channels}, got {x.shape[-1]}.")

        kh, kw = self.kernel_size
        x_pad = pad_2d(x, kh // 2, kw // 2, self.padding_mode)

        return lax.conv_general_dilated(
            lhs=x_pad,
            rhs=self.kernel,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.groups,
        )

    def activation(self, x: Array) -> Array:
        """Apply sign activation function.

        Parameters
        ----------
        x : Array
            Input tensor.

        Returns
        -------
        Array
            Sign of input: -1 for negative, +1 for positive, 0 for zero.

        """
        return jnp.sign(x)

    def reduce(self, h: PyTree) -> Array:
        """Reduce a pytree by summing all leaves.

        Parameters
        ----------
        h : PyTree
            Tree structure to reduce.

        Returns
        -------
        Array
            Sum of all leaf values.

        """
        return jnp.asarray(tree_reduce(operator.add, h))

    # ---------- backward ----------
    def _diag_group_blocks(self, dw_full: Array) -> Array:
        """Extract within-group diagonal blocks from full gradient.

        Parameters
        ----------
        dw_full : Array
            Full gradient of shape (Kh, Kw, Cin, Cout) where
            Cin = channels and Cout = channels.

        Returns
        -------
        Array
            Diagonal blocks of shape (Kh, Kw, cin_g, Cout) containing
            only within-group connections.

        Notes
        -----
        For grouped convolutions, we only want updates within each group.
        This function extracts the diagonal blocks corresponding to
        within-group connections and discards cross-group gradients.

        Example:
            For groups=2, channels=4 (cin_g=2):
            - Group 0: input channels 0-1 connect to output channels 0-1
            - Group 1: input channels 2-3 connect to output channels 2-3
            Cross-group connections (e.g., input 0 to output 2) are discarded.

        """
        kh, kw = self.kernel_size
        g = self.groups
        cin_g = self.channels // g
        cout_g = self.channels // g

        # (kh,kw, Cin, Cout) -> (kh,kw, g, cin_g, g, cout_g)
        dW6 = dw_full.reshape(kh, kw, g, cin_g, g, cout_g)

        # take diagonal over the two group axes -> (kh,kw, cin_g, cout_g, g)
        diag = jnp.diagonal(dW6, axis1=2, axis2=4)

        # move g next to cout_g and flatten to Cout -> (kh,kw, cin_g, Cout)
        return jnp.transpose(diag, (0, 1, 2, 4, 3)).reshape(kh, kw, cin_g, self.channels)

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Compute weight update with diagonal constraint.

        Parameters
        ----------
        x : Array
            Input tensor of shape (N, H, W, channels).
        y : Array
            Primary output signal of shape (N, H, W, channels).
        y_hat : Array
            Secondary signal for gating of shape (N, H, W, channels).
        gate : Array or None, default=None
            Optional global gate (unused).

        Returns
        -------
        Self
            Update tree with kernel update. The diagonal self-connections
            at the central element are masked to remain at j_d.

        Notes
        -----
        Update Computation:
            1. Compute gradient using conv_backward_with_threshold
            2. Extract within-group diagonal blocks
            3. Add weight decay term
            4. Apply update_mask to zero out constrained parameters

        Constraint Enforcement:
            The update_mask ensures that diagonal entries at the central
            kernel element receive zero gradient, preserving j_d forever.

        Normalization:
            Both gradient and weight decay normalized by 1/sqrt(N × Ho × Wo).

        """
        kh, kw = self.kernel_size

        # conv_utils returns (Kh, Kw, Cin, Cout) with normalization by 1/sqrt(N × Ho × Wo)
        dw_full = conv_backward_with_threshold(
            x=x,
            y=y,
            y_hat=y_hat,
            threshold=self.threshold,
            kernel_shape=(kh, kw),
            groups=self.groups,
            strides=(1, 1),
            padding_mode=self.padding_mode,
        )

        dW = self._diag_group_blocks(dw_full)  # (Kh, Kw, cin_g, Cout)

        # Weight decay with same normalization as gradient: 1/sqrt(N × Ho × Wo)
        n, ho, wo, _ = y.shape
        decay_scale = 1.0 / jnp.sqrt(jnp.asarray(n * ho * wo, dtype=dW.dtype))

        dW = self.lr * dW + self.weight_decay * decay_scale * self.kernel

        # hard constraint: prevent any change to constrained params (includes decay term)
        dW = dW * self.update_mask

        zero_update: Self = jax.tree_util.tree_map(
            jnp.zeros_like, self, is_leaf=eqx.is_inexact_array
        )
        update: Self = eqx.tree_at(lambda m: m.kernel, zero_update, dW)
        return update


class Conv2DTranspose(Adapter):
    """Transposed convolution (deconvolution) with Hebbian-style learning.

    This adapter implements a transposed 2D convolution (also known as
    deconvolution or upsampling convolution) with activity-dependent weight
    updates similar to Conv2D.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple[int, int]
        Size of the convolution kernel. If int, same size for both dimensions.
    threshold : float
        Threshold for gating updates. Positions where
        (y * y_hat < threshold) receive updates.
    strength : float
        Multiplicative factor applied to output.
    key : KeyArray
        JAX random key for weight initialization.
    stride : int or tuple[int, int], default=1
        Upsampling factor (implemented via rhs_dilation).
    padding_mode : str, callable, or None, default=None
        Padding mode ('constant', 'edge', 'reflect', or None).
    dtype : DTypeLike, default=jnp.float32
        Data type for parameters.

    Attributes
    ----------
    kernel : Array
        Convolution kernel of shape (Kh, Kw, in_channels, out_channels).
    threshold : Array
        Scalar threshold value.
    strength : float
        Output scaling factor (static).
    in_channels : int
        Number of input channels (static).
    out_channels : int
        Number of output channels (static).
    stride : int
        Upsampling factor stored as scalar (static).
    kernel_size : tuple[int, int]
        Kernel spatial size (static).
    padding_mode : str, callable, or None
        Padding mode (static).

    Notes
    -----
    Weight Initialization:
        Kernels are initialized from N(0, 1/fan_in) where fan_in = Kh × Kw × Cin.

    Transposed Convolution:
        Implemented using rhs_dilation to achieve upsampling effect.
        For stride=2, output size is approximately 2× input size.

    Learning:
        This version returns only the raw gradient from
        conv_transpose_backward_with_threshold without explicit learning rate
        or weight decay terms applied in the backward method.

    Examples
    --------
    >>> key = jax.random.PRNGKey(0)
    >>> deconv = Conv2DTranspose(
    ...     in_channels=64,
    ...     out_channels=32,
    ...     kernel_size=3,
    ...     threshold=0.0,
    ...     strength=1.0,
    ...     key=key,
    ...     stride=2
    ... )
    >>> x = jax.random.normal(key, (1, 16, 16, 64))
    >>> y = deconv(x)
    >>> y.shape  # Upsampled by factor of 2
    (1, 32, 32, 32)

    """

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
        """Initialize Conv2DTranspose adapter with proper variance scaling."""
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        stride_tuple = fetch_tuple_from_arg(stride)
        self.stride = int(stride_tuple[0])
        self.kernel_size = fetch_tuple_from_arg(kernel_size)
        self.padding_mode = padding_mode
        self.threshold = jnp.asarray(threshold, dtype=dtype)
        self.strength = float(strength)

        key, init_key = jax.random.split(key)
        kh, kw = self.kernel_size
        fan_in = kh * kw * self.in_channels
        self.kernel = jax.random.normal(
            key=init_key,
            shape=(kh, kw, self.in_channels, self.out_channels),
            dtype=dtype,
        ) / jnp.sqrt(jnp.asarray(fan_in, dtype=dtype))

    def __call__(self, x: Array, rng: KeyArray | None = None) -> Array:
        """Forward pass through transposed convolution.

        Parameters
        ----------
        x : Array
            Input tensor of shape (N, H, W, Cin).
        rng : KeyArray or None, default=None
            Random key (unused, for interface compatibility).

        Returns
        -------
        Array
            Output tensor of shape (N, Ho, Wo, Cout) where Ho and Wo are
            upsampled by the stride factor.

        """
        return conv_transpose_forward(x, self.kernel, self.stride, self.padding_mode)

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        """Compute weight update for transposed convolution.

        Parameters
        ----------
        x : Array
            Input tensor of shape (N, H, W, Cin).
        y : Array
            Primary output signal of shape (N, Ho, Wo, Cout).
        y_hat : Array
            Secondary signal for gating of shape (N, Ho, Wo, Cout).
        gate : Array or None, default=None
            Optional global gate (unused).

        Returns
        -------
        Self
            Update tree with kernel update.

        Notes
        -----
        This implementation returns the raw gradient without applying
        learning rate or weight decay. These should be handled externally
        if needed.

        """
        dW = conv_transpose_backward_with_threshold(
            x=x,
            y=y,
            y_hat=y_hat,
            threshold=self.threshold,
            kernel_shape=self.kernel_size,
            stride=self.stride,
            padding_mode=self.padding_mode,
        )
        zero_update: Self = jax.tree_util.tree_map(
            jnp.zeros_like, self, is_leaf=eqx.is_inexact_array
        )
        update: Self = eqx.tree_at(lambda m: m.kernel, zero_update, dW)
        return update
