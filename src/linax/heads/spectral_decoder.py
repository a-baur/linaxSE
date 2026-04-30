"""SEMamba-style mag/phase decoders.

Each head consumes the block-stack output ``[C, T, F/2]`` and:

1. Upsamples the freq axis with a 1-D conv-transpose along F.
2. Applies InstanceNorm (via per-channel ``GroupNorm``) and PReLU.
3. Collapses the channel axis with one or more 1×1 convs.

The mag head adds a sigmoid to produce a bounded multiplicative mask. The
phase head ends in two parallel 1×1 conv branches (pseudo-real and pseudo-imag)
whose ratio is fed into ``atan2``, so the result lives in ``(-π, π]``.

Default kernel/stride invert the ``DenseEncoder.dense_conv_2`` reduction
(kernel ``(1, 3)``, stride ``(1, 2)``, padding 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from linax.heads.base import Head, HeadConfig
from linax.modules import PReLU


def _build_upsample(
    in_channels: int,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    key: PRNGKeyArray,
) -> tuple[eqx.nn.ConvTranspose2d, eqx.nn.GroupNorm, PReLU]:
    """Construct the (ConvTranspose2d, InstanceNorm, PReLU) trio shared by both heads."""
    conv = eqx.nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=in_channels,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        key=key,
    )
    norm = eqx.nn.GroupNorm(
        groups=in_channels, channels=in_channels, channelwise_affine=True
    )
    act = PReLU(in_channels)
    return conv, norm, act


@dataclass(frozen=True)
class MagDecoderHeadConfig(HeadConfig):
    """Magnitude-mask decoder.

    Attributes:
        upsample_kernel: Conv-transpose kernel along (T, F). The default
            ``(1, 3)`` undoes the dense encoder's ``(1, 3)`` reduction.
        upsample_stride: Conv-transpose stride along (T, F). The default
            ``(1, 2)`` undoes the encoder's stride-2 freq reduction.
        upsample_padding: Conv-transpose padding.
        out_features: Required by ``HeadConfig``; ignored. The output freq dim
            is determined by the conv-transpose, not this field.
        reduce: Required by ``HeadConfig``; ignored.
    """

    out_features: int = 0
    reduce: bool = False
    upsample_kernel: tuple[int, int] = (1, 3)
    upsample_stride: tuple[int, int] = (1, 2)
    upsample_padding: tuple[int, int] = (0, 0)

    def build(self, in_features: int, key: PRNGKeyArray) -> MagDecoderHead:
        """Build head from config."""
        return MagDecoderHead(in_features=in_features, cfg=self, key=key)


class MagDecoderHead(Head):
    """ConvTranspose upsample → InstanceNorm → PReLU → 1×1 conv → sigmoid."""

    upsample: eqx.nn.ConvTranspose2d
    norm: eqx.nn.GroupNorm
    activation: PReLU
    project: eqx.nn.Conv2d

    def __init__(
        self, in_features: int, cfg: MagDecoderHeadConfig, key: PRNGKeyArray
    ):
        """Initialise the magnitude decoder head."""
        u_key, p_key = jr.split(key, 2)
        self.upsample, self.norm, self.activation = _build_upsample(
            in_channels=in_features,
            kernel=cfg.upsample_kernel,
            stride=cfg.upsample_stride,
            padding=cfg.upsample_padding,
            key=u_key,
        )
        self.project = eqx.nn.Conv2d(
            in_channels=in_features,
            out_channels=1,
            kernel_size=(1, 1),
            key=p_key,
        )

    def __call__(
        self,
        x: Float[Array, "channels frames bins_in"],
        state: eqx.nn.State,
    ) -> tuple[Float[Array, "frames bins_out"], eqx.nn.State]:
        """Forward pass. Output is in ``[0, 1]`` per (frame, freq)."""
        h = self.upsample(x)  # [C, T, F]
        h = self.norm(h)
        h = self.activation(h)
        h = self.project(h)  # [1, T, F]
        h = jax.nn.sigmoid(h)
        return h.squeeze(0), state  # [T, F]


@dataclass(frozen=True)
class PhaseDecoderHeadConfig(HeadConfig):
    """Phase decoder.

    Same upsample/norm/activation pipeline as the mag head, but ends in two
    parallel 1×1 conv branches whose outputs are combined via
    ``atan2(imag, real)``.
    """

    out_features: int = 0
    reduce: bool = False
    upsample_kernel: tuple[int, int] = (1, 3)
    upsample_stride: tuple[int, int] = (1, 2)
    upsample_padding: tuple[int, int] = (0, 0)

    def build(self, in_features: int, key: PRNGKeyArray) -> PhaseDecoderHead:
        """Build head from config."""
        return PhaseDecoderHead(in_features=in_features, cfg=self, key=key)


class PhaseDecoderHead(Head):
    """ConvTranspose upsample → InstanceNorm → PReLU → atan2(imag, real)."""

    upsample: eqx.nn.ConvTranspose2d
    norm: eqx.nn.GroupNorm
    activation: PReLU
    project_real: eqx.nn.Conv2d
    project_imag: eqx.nn.Conv2d

    def __init__(
        self, in_features: int, cfg: PhaseDecoderHeadConfig, key: PRNGKeyArray
    ):
        """Initialise the phase decoder head."""
        u_key, r_key, i_key = jr.split(key, 3)
        self.upsample, self.norm, self.activation = _build_upsample(
            in_channels=in_features,
            kernel=cfg.upsample_kernel,
            stride=cfg.upsample_stride,
            padding=cfg.upsample_padding,
            key=u_key,
        )
        self.project_real = eqx.nn.Conv2d(
            in_channels=in_features,
            out_channels=1,
            kernel_size=(1, 1),
            key=r_key,
        )
        self.project_imag = eqx.nn.Conv2d(
            in_channels=in_features,
            out_channels=1,
            kernel_size=(1, 1),
            key=i_key,
        )

    def __call__(
        self,
        x: Float[Array, "channels frames bins_in"],
        state: eqx.nn.State,
    ) -> tuple[Float[Array, "frames bins_out"], eqx.nn.State]:
        """Forward pass. Output is the wrapped phase in ``(-π, π]``."""
        h = self.upsample(x)  # [C, T, F]
        h = self.norm(h)
        h = self.activation(h)
        x_r = self.project_real(h).squeeze(0)  # [T, F]
        x_i = self.project_imag(h).squeeze(0)  # [T, F]
        return jnp.arctan2(x_i, x_r), state
