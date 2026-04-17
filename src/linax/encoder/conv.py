"""Convolutional encoder."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from linax.encoder.base import Encoder, EncoderConfig


@dataclass(frozen=True)
class ConvEncoderConfig(EncoderConfig):
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
    use_bias: bool = False

    def build(self, key: PRNGKeyArray) -> ConvEncoder:
        """Build encoder from config.

        Args:
            key: JAX random key for initialization.

        Returns:
            The encoder instance.
        """
        assert len(self.in_dims) == len(self.out_dims) == len(self.kernel_size)
        return ConvEncoder(cfg=self, key=key)


class ConvEncoder[ConfigType: ConvEncoderConfig](Encoder):
    """Convolutional encoder.

    This encoder takes an input of shape (timesteps, in_features)
    and outputs a hidden representation of shape (timesteps, hidden_dim).

    Args:
        cfg: Configuration for the convolutional encoder.
        key: JAX random key for initialization.

    Attributes:
        layers: List of convolutional layers.
    """

    layers: eqx.nn.Sequential

    def __init__(self, cfg: ConfigType, key: PRNGKeyArray):
        """Initialize the conv encoder."""
        keys = jax.random.split(key, len(cfg.in_dims))

        self.layers = eqx.nn.Sequential(
            [
                eqx.nn.Conv1d(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    stride=1,
                    key=k,
                    use_bias=cfg.use_bias,
                )
                for in_dim, out_dim, kernel_size, k in zip(
                    cfg.in_dims, cfg.out_dims, cfg.kernel_size, keys
                )
            ]
        )

    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the conv encoder.

        Args:
            x: Input tensor.
            state: Current state for stateful layers.

        Returns:
            Tuple containing the output tensor and updated state.
        """
        x = jax.numpy.swapaxes(x, 0, 1)
        x = self.layers(x)
        x = jax.numpy.swapaxes(x, 0, 1)
        return x, state
