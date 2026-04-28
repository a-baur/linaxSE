"""Convolutional encoder."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from linax.encoder.base import Encoder, EncoderConfig


@dataclass(frozen=True)
class ChunkedConvEncoderConfig(EncoderConfig):
    """Configuration for the convolutional encoder.

    Attributes:
        in_features: Input dimensionality.
        out_features: Output dimensionality .
        use_bias: Whether to use bias in the conv layers.
    """

    in_features: int
    out_features: int
    in_dims: tuple[int, ...]
    out_dims: tuple[int, ...]
    kernel_size: tuple[int, ...]
    causal: bool = False
    use_bias: bool = False

    def build(self, key: PRNGKeyArray) -> ConvEncoder:
        """Build encoder from config.

        Args:
            key: JAX random key for initialization.

        Returns:
            The encoder instance.
        """
        assert len(self.in_dims) == len(self.out_dims) == len(self.kernel_size)
        return ChunkedConvEncoder(cfg=self, key=key)


class ChunkedConvEncoder[ConfigType: ChunkedConvEncoderConfig](Encoder):
    """Convolutional encoder."""

    layers: eqx.nn.Sequential
    causal: bool

    def __init__(self, cfg: ConfigType, key: PRNGKeyArray):
        """Initialize the conv encoder."""
        keys = jax.random.split(key, len(cfg.in_dims))

        self.causal = cfg.causal

        layers_list = []
        for in_dim, out_dim, kernel_size, k in zip(
                cfg.in_dims, cfg.out_dims, cfg.kernel_size, keys
        ):
            if self.causal:
                # Pad only the 'past'
                pad = [(kernel_size - 1, 0)]
            else:
                # Symmetric padding for non-causal
                pad = (kernel_size - 1) // 2

            layers_list.append(
                eqx.nn.Conv1d(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    padding=pad,
                    stride=1,
                    key=k,
                    use_bias=cfg.use_bias,
                )
            )

        self.layers = eqx.nn.Sequential(layers_list)

    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the conv encoder."""
        timesteps, channels = x.shape
        chunk_size = 64

        num_chunks = timesteps // chunk_size

        x_chunked = x[:num_chunks * chunk_size, :]
        x_chunked = x_chunked.reshape(num_chunks, chunk_size, channels)

        # Conv1d expects (channels, length); chunks are (length, channels).
        x_chunked = jax.numpy.swapaxes(x_chunked, 1, 2)
        x_chunked = jax.vmap(self.layers)(x_chunked)
        x_chunked = jax.numpy.swapaxes(x_chunked, 1, 2)

        out_channels = x_chunked.shape[-1]
        x_out = x_chunked.reshape(-1, out_channels)
        return x_out, state


@dataclass(frozen=True)
class ConvEncoderConfig(EncoderConfig):
    """Configuration for the convolutional encoder.

    Attributes:
        in_features: Input dimensionality.
        out_features: Output dimensionality .
        dilation: Per-layer dilation. Defaults to all 1s when None.
        use_bias: Whether to use bias in the conv layers.
    """

    in_features: int
    out_features: int
    in_dims: tuple[int, ...]
    out_dims: tuple[int, ...]
    kernel_size: tuple[int, ...]
    dilation: tuple[int, ...] | None = None
    causal: bool = False
    use_bias: bool = False

    def build(self, key: PRNGKeyArray) -> ConvEncoder:
        """Build encoder from config.

        Args:
            key: JAX random key for initialization.

        Returns:
            The encoder instance.
        """
        assert len(self.in_dims) == len(self.out_dims) == len(self.kernel_size)
        if self.dilation is not None:
            assert len(self.dilation) == len(self.in_dims)
        return ConvEncoder(cfg=self, key=key)


class ConvEncoder[ConfigType: ConvEncoderConfig](Encoder):
    """Convolutional encoder."""

    layers: eqx.nn.Sequential
    causal: bool

    def __init__(self, cfg: ConfigType, key: PRNGKeyArray):
        """Initialize the conv encoder."""
        keys = jax.random.split(key, len(cfg.in_dims))

        self.causal = cfg.causal

        dilations = cfg.dilation if cfg.dilation is not None else (1,) * len(cfg.in_dims)

        layers_list = []
        for in_dim, out_dim, kernel_size, dilation, k in zip(
                cfg.in_dims, cfg.out_dims, cfg.kernel_size, dilations, keys
        ):
            effective_kernel = (kernel_size - 1) * dilation
            if self.causal:
                # Pad only the 'past' (scaled by dilation)
                pad = [(effective_kernel, 0)]
            else:
                # Symmetric padding for non-causal
                pad = effective_kernel // 2

            layers_list.append(
                eqx.nn.Conv1d(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    padding=pad,
                    stride=1,
                    dilation=dilation,
                    key=k,
                    use_bias=cfg.use_bias,
                )
            )

        self.layers = eqx.nn.Sequential(layers_list)

    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the conv encoder."""
        x = jax.numpy.swapaxes(x, 0, 1)
        x = self.layers(x)
        x = jax.numpy.swapaxes(x, 0, 1)

        return x, state
