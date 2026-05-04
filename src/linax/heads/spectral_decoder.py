"""SEMamba-style mag/phase decoders.

Each head consumes the block-stack output ``[C, T, F_in]`` and:

1. Upsamples the freq axis using one of two strategies (``upsample_type``):

   - ``resize_conv``: nearest-neighbour resize to ``target_freq`` followed by
     a regular ``Conv2d``. Routes through cuDNN's forward-conv path with
     well-tuned tensor-core kernels.
   - ``transposed``: ``ConvTranspose2d(kernel, stride, padding)`` — matches
     SEMamba's reference implementation. XLA expresses the backward weight
     gradient as a giant-kernel forward conv, which lands on cuDNN's slower
     algorithms.

2. Applies InstanceNorm (via per-channel ``GroupNorm``) and PReLU.
3. Collapses the channel axis with one or more 1×1 convs.

The mag head adds a sigmoid to produce a bounded multiplicative mask. The
phase head ends in two parallel 1×1 conv branches (pseudo-real and
pseudo-imag) whose ratio is fed into ``atan2``, so the result lives in
``(-π, π]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from linax.heads.base import Head, HeadConfig
from linax.modules import PReLU

UpsampleType = Literal["resize_conv", "transposed"]


def _build_upsample(
    in_channels: int,
    upsample_type: UpsampleType,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    key: PRNGKeyArray,
) -> eqx.Module:
    """Build the upsample conv for the requested strategy.

    For ``resize_conv`` the spatial upsample is done outside (via
    ``jax.image.resize`` in ``__call__``); the conv only refines the result
    and uses freq pad of 1 with kernel ``(1, 3)`` to preserve shape.
    For ``transposed`` the conv handles upsampling itself via stride.
    """
    if upsample_type == "resize_conv":
        return eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel,
            padding=(0, 1),
            key=key,
        )
    if upsample_type == "transposed":
        return eqx.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            key=key,
        )
    raise ValueError(f"unknown upsample_type: {upsample_type!r}")


@dataclass(frozen=True)
class MagDecoderHeadConfig(HeadConfig):
    """Magnitude-mask decoder.

    Attributes:
        upsample_type: ``resize_conv`` (NN-resize + Conv2d, fast on JAX/cuDNN)
            or ``transposed`` (ConvTranspose2d, SEMamba-faithful).
        target_freq: Output freq dim. Used by ``resize_conv`` to size the
            resize. For ``transposed`` you must pick ``upsample_stride`` /
            ``upsample_padding`` so the conv-transpose lands on this size.
        upsample_kernel: Conv kernel along (T, F).
        upsample_stride: ``transposed`` only.
        upsample_padding: ``transposed`` only.
        out_features / reduce: Required by ``HeadConfig``; ignored.
    """

    out_features: int = 0
    reduce: bool = False
    upsample_type: UpsampleType = "resize_conv"
    target_freq: int = 0
    upsample_kernel: tuple[int, int] = (1, 3)
    upsample_stride: tuple[int, int] = (1, 2)
    upsample_padding: tuple[int, int] = (0, 0)

    def build(self, in_features: int, key: PRNGKeyArray) -> MagDecoderHead:
        """Build head from config."""
        return MagDecoderHead(in_features=in_features, cfg=self, key=key)


class MagDecoderHead(Head):
    """Upsample → InstanceNorm → PReLU → 1×1 conv → sigmoid."""

    upsample: eqx.Module
    norm: eqx.nn.GroupNorm
    activation: PReLU
    project: eqx.nn.Conv2d
    target_freq: int = eqx.field(static=True)
    upsample_type: UpsampleType = eqx.field(static=True)

    def __init__(self, in_features: int, cfg: MagDecoderHeadConfig, key: PRNGKeyArray):
        """Initialise the magnitude decoder head."""
        u_key, p_key = jr.split(key, 2)
        self.upsample_type = cfg.upsample_type
        self.target_freq = cfg.target_freq
        self.upsample = _build_upsample(
            in_channels=in_features,
            upsample_type=cfg.upsample_type,
            kernel=cfg.upsample_kernel,
            stride=cfg.upsample_stride,
            padding=cfg.upsample_padding,
            key=u_key,
        )
        self.norm = eqx.nn.GroupNorm(
            groups=in_features, channels=in_features, channelwise_affine=True
        )
        self.activation = PReLU(in_features)
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
        if self.upsample_type == "resize_conv":
            c, t, _ = x.shape
            h = jax.image.resize(x, (c, t, self.target_freq), method="nearest")
            h = self.upsample(h)
        else:
            h = self.upsample(x)
        h = self.norm(h)
        h = self.activation(h)
        h = self.project(h)
        h = jax.nn.sigmoid(h)
        return h.squeeze(0), state


@dataclass(frozen=True)
class PhaseDecoderHeadConfig(HeadConfig):
    """Phase decoder.

    Same upsample/norm/activation pipeline as the mag head, but ends in two
    parallel 1×1 conv branches whose outputs are combined via
    ``atan2(imag, real)``.
    """

    out_features: int = 0
    reduce: bool = False
    upsample_type: UpsampleType = "resize_conv"
    target_freq: int = 0
    upsample_kernel: tuple[int, int] = (1, 3)
    upsample_stride: tuple[int, int] = (1, 2)
    upsample_padding: tuple[int, int] = (0, 0)

    def build(self, in_features: int, key: PRNGKeyArray) -> PhaseDecoderHead:
        """Build head from config."""
        return PhaseDecoderHead(in_features=in_features, cfg=self, key=key)


class PhaseDecoderHead(Head):
    """Upsample → InstanceNorm → PReLU → atan2(imag, real)."""

    upsample: eqx.Module
    norm: eqx.nn.GroupNorm
    activation: PReLU
    project_real: eqx.nn.Conv2d
    project_imag: eqx.nn.Conv2d
    target_freq: int = eqx.field(static=True)
    upsample_type: UpsampleType = eqx.field(static=True)

    def __init__(self, in_features: int, cfg: PhaseDecoderHeadConfig, key: PRNGKeyArray):
        """Initialise the phase decoder head."""
        u_key, r_key, i_key = jr.split(key, 3)
        self.upsample_type = cfg.upsample_type
        self.target_freq = cfg.target_freq
        self.upsample = _build_upsample(
            in_channels=in_features,
            upsample_type=cfg.upsample_type,
            kernel=cfg.upsample_kernel,
            stride=cfg.upsample_stride,
            padding=cfg.upsample_padding,
            key=u_key,
        )
        self.norm = eqx.nn.GroupNorm(
            groups=in_features, channels=in_features, channelwise_affine=True
        )
        self.activation = PReLU(in_features)
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
        if self.upsample_type == "resize_conv":
            c, t, _ = x.shape
            h = jax.image.resize(x, (c, t, self.target_freq), method="nearest")
            h = self.upsample(h)
        else:
            h = self.upsample(x)
        h = self.norm(h)
        h = self.activation(h)
        x_r = self.project_real(h).squeeze(0)
        x_i = self.project_imag(h).squeeze(0)
        return jnp.arctan2(x_i, x_r), state
