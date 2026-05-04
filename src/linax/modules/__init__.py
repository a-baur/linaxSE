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
            self.dense_layers.append(
                DenseConv(
                    in_channels=hidden_size,
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
