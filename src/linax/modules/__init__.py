"""Common modules for all models."""

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PRNGKeyArray


def get_padding(kernel_size, dilation=1):
    """Calculate the padding size for a convolutional layer.

    Args:
    - kernel_size (int): Size of the convolutional kernel.
    - dilation (int, optional): Dilation rate of the convolution. Defaults to 1.

    Returns:
    - int: Calculated padding size.
    """
    return int((kernel_size * dilation - dilation) / 2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    """Calculate the padding size for a 2D convolutional layer.

    Args:
    - kernel_size (tuple): Size of the convolutional kernel (height, width).
    - dilation (tuple, optional): Dilation rate of the convolution (height, width). Defaults to (1, 1).

    Returns:
    - tuple: Calculated padding size (height, width).
    """
    return (
        int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
        int((kernel_size[1] * dilation[1] - dilation[1]) / 2),
    )


class PReLU(eqx.Module):
    weight: jax.Array

    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        """Args:
        num_parameters: 1 for a single shared slope, or `channels` for channel-wise.
        init: Initial value for the negative slope.
        """
        self.weight = jnp.full((num_parameters,), init)

    def __call__(self, x: jax.Array) -> jax.Array:
        weight = self.weight

        # If weight is per-channel (C,) and input is (C, Freq, Time),
        # we must reshape weight to (C, 1, 1) so it multiplies correctly.
        if weight.shape[0] > 1 and x.ndim > 1:
            broadcast_shape = (weight.shape[0],) + (1,) * (x.ndim - 1)
            weight = weight.reshape(broadcast_shape)

        return jnp.where(x >= 0, x, weight * x)


class RMSNorm(eqx.Module):
    """Root-mean-square LayerNorm: ``y = x / sqrt(mean(x**2) + eps) * weight``.

    Per-feature learnable gain, no mean subtraction, no bias — matches the
    ``mamba_ssm.ops.triton.layernorm.RMSNorm`` used by SEMamba's ``Block``
    (with ``norm_epsilon=1e-5`` per the SEMamba_advanced YAML).
    """

    weight: jax.Array
    eps: float = eqx.field(static=True)

    def __init__(self, shape: int, eps: float = 1e-5):
        self.weight = jnp.ones((shape,))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


class LearnableSigmoid2d(eqx.Module):
    """SEMamba-style learnable-slope sigmoid: ``beta * sigmoid(slope * x)``.

    ``slope`` is a learnable per-feature vector initialised to ones; ``beta``
    is a fixed scalar hyperparameter. SEMamba parameterises the slope by the
    frequency dim with shape ``(F, 1)`` over a ``(B, F, T)`` input; the linax
    decoders carry ``(C, T, F)`` so we store the slope as ``(F,)`` and let it
    broadcast across the channel and time axes — semantically the same
    per-frequency learnable slope.
    """

    slope: jax.Array
    beta: float = eqx.field(static=True)

    def __init__(self, in_features: int, beta: float = 1.0):
        self.slope = jnp.ones((in_features,))
        self.beta = beta

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.beta * jax.nn.sigmoid(self.slope * x)


class DenseConv(eqx.Module):
    conv: nn.Conv2d
    instance_norm: nn.GroupNorm
    activation: eqx.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        dilation: tuple = (1, 1),
        padding: tuple = (0, 0),
        stride: tuple = (1, 1),
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
            key=key,
        )
        self.instance_norm = nn.GroupNorm(
            groups=out_channels,
            channels=out_channels,
            channelwise_affine=True,
        )
        self.activation = PReLU(out_channels)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.activation(x)
        return x


class DenseBlock(eqx.Module):
    """Stack of dilated convs with residual connections.

    Each layer is a dilated conv with the same in/out channel count; its
    output is added to the running activation. Originally a DenseNet-style
    block (channels grew via concatenation), but the dense connections
    multiplied activation memory and made the deepest conv bandwidth-bound.
    """

    num_layers: int
    hidden_size: int
    dense_layers: list
    skip_type: Literal["residual", "dense"]

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        kernel_size: tuple = (3, 3),
        dilation_rate: int = 2,
        skip_type: Literal["residual", "dense"] = "dense",
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.skip_type = skip_type
        self.dense_layers = []

        keys = jax.random.split(key, num_layers)

        for i in range(num_layers):
            d = dilation_rate**i
            # Dense (concat) skip grows the running buffer to (i+1)*hidden along
            # the channel axis on iter i, so each conv must accept the wider
            # input. Residual skip keeps the channel count fixed at hidden.
            in_ch = hidden_size * (i + 1) if skip_type == "dense" else hidden_size
            self.dense_layers.append(
                DenseConv(
                    in_channels=in_ch,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=(d, 1),
                    stride=(1, 1),
                    padding=get_padding_2d(kernel_size, (d, 1)),
                    key=keys[i],
                )
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        skip = x
        for layer in self.dense_layers:
            if self.skip_type == "residual":
                x = skip + layer(x)
            elif self.skip_type == "dense":
                x = layer(skip)
                skip = jnp.concatenate((x, skip), axis=0)
        return x


class FANLayer(eqx.Module):
    """Fourier Analysis Network."""

    in_features: int
    out_features: int

    W_p: jax.Array
    W_p_bar: jax.Array
    B_p_bar: jax.Array

    def __init__(self, in_features: int, out_features: int, key: PRNGKeyArray) -> None:
        self.in_features = in_features
        self.out_features = out_features

        d_p = out_features // 4
        d_p_bar = out_features - 2 * d_p

        k1, k2 = jax.random.split(key, 2)
        initializer = jax.nn.initializers.glorot_uniform()
        self.W_p = initializer(k1, (in_features, d_p), jnp.float32)
        self.W_p_bar = initializer(k2, (in_features, d_p_bar), jnp.float32)
        self.B_p_bar = jnp.zeros((d_p_bar,), dtype=jnp.float32)

    def __call__(self, x: jax.Array) -> jax.Array:
        cos_term = jnp.cos(x @ self.W_p)
        sin_term = jnp.sin(x @ self.W_p)
        lin_term = jax.nn.gelu((x @ self.W_p_bar) + self.B_p_bar)

        output = jnp.concatenate((cos_term, sin_term, lin_term), axis=-1)
        return output
