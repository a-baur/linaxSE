"""This module contains the encoders implemented in Linax."""

from linax.encoder.base import Encoder, EncoderConfig
from linax.encoder.conv import ConvEncoder, ConvEncoderConfig
from linax.encoder.embedding import (
    EmbeddingEncoder,
    EmbeddingEncoderConfig,
)
from linax.encoder.linear import LinearEncoder, LinearEncoderConfig

__all__ = [
    "ConvEncoderConfig",
    "ConvEncoder",
    "EncoderConfig",
    "Encoder",
    "LinearEncoder",
    "LinearEncoderConfig",
    "EmbeddingEncoder",
    "EmbeddingEncoderConfig",
]
