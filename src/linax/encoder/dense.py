"""Convolutional encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from linax.encoder.base import Encoder, EncoderConfig
from linax.modules import DenseBlock, DenseConv


@dataclass(frozen=True)
class DenseEncoderConfig(EncoderConfig):
    """Configuration for the convolutional encoder.

    Attributes:
        in_channels: Input dimensionality.
        out_channels: Output dimensionality .
        dense_layers: Number of dense layers.
    """

    in_channels: int
    out_channels: int
    dense_layers: int
    skip_type: Literal["residual", "dense"] = "dense"

    def build(self, key: PRNGKeyArray) -> DenseEncoder:
        """Build encoder from config.

        Args:
            key: JAX random key for initialization.

        Returns:
            The encoder instance.
        """
        return DenseEncoder(cfg=self, key=key)


class DenseEncoder[ConfigType: DenseEncoderConfig](Encoder):
    """Dense encoder."""

    dense_conv_1: DenseConv
    dense_block: DenseBlock
    dense_conv_2: DenseConv

    def __init__(self, cfg: ConfigType, key: PRNGKeyArray):
        """Initialize the conv encoder."""
        conv1_key, block_key, conv2_key = jax.random.split(key, 3)
        self.dense_conv_1 = DenseConv(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            kernel_size=(1, 1),
            key=conv1_key,
        )
        self.dense_block = DenseBlock(
            num_layers=cfg.dense_layers,
            hidden_size=cfg.out_channels,
            kernel_size=(3, 3),
            dilation_rate=1,
            skip_type=cfg.skip_type,
            key=block_key,
        )
        self.dense_conv_2 = DenseConv(
            in_channels=cfg.out_channels,
            out_channels=cfg.out_channels,
            kernel_size=(1, 3),
            stride=(1, 2),
            key=conv2_key,
        )

    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        """Forward pass of the conv encoder."""
        x = self.dense_conv_1(x)  # [out_features, frames, bins]
        x = self.dense_block(x)  # [out_features, frames, bins]
        x = self.dense_conv_2(x)  # [out_features, frames, bins // 2]
        return x, state
