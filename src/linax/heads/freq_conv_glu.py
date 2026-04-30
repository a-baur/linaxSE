"""Convolutional head with stacked GLU-gated 1-D convs along the frequency axis."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from linax.heads.base import Head, HeadConfig


@dataclass(frozen=True)
class FreqConvGLUHeadConfig(HeadConfig):
    """Configuration for the convolutional frequency-recurrent head.

    A stack of 1-D convs runs along the feature (frequency) axis of each
    frame; each layer produces a value/gate pair combined with GLU, keeping
    the signal at a single channel between layers. With ``causal=True``
    each output bin depends only on equal-or-lower-indexed input bins,
    mimicking a recurrent sweep up the frequency axis.

    Attributes:
        out_features: Output feature dimensionality.
        kernel_size: Kernel size of the conv along the frequency axis.
        num_layers: Number of stacked GLU-gated convs.
        dilation: Per-layer dilation along the frequency axis. Defaults to
            all 1s when None; otherwise must have length ``num_layers``.
        causal: If True, pad on the low-frequency side only.
        use_bias: Whether the conv and output projection use bias.
    """

    kernel_size: int = 5
    num_layers: int = 1
    dilation: tuple[int, ...] | None = None
    causal: bool = True
    use_bias: bool = True


    def build(self, in_features: int, key: PRNGKeyArray) -> FreqConvGLUHead:
        """Build head from config."""
        return FreqConvGLUHead(in_features=in_features, cfg=self, key=key)


def _padding(kernel_size: int, dilation: int, causal: bool) -> int | list[tuple[int, int]]:
    """Padding spec that keeps the spatial (frequency) size unchanged."""
    effective_kernel = (kernel_size - 1) * dilation
    if causal:
        return [(effective_kernel, 0)]
    return effective_kernel // 2


class FreqConvGLUHead[ConfigType: FreqConvGLUHeadConfig](Head):
    """Convolutional head with stacked GLU-gated convs along the frequency axis.

    Each time frame is treated as a 1-channel 1-D signal of length
    ``in_features``. At every layer a conv with two output channels yields
    the value and gate, which are combined as ``value * sigmoid(gate)``,
    returning a single channel for the next layer. A per-frame Linear maps
    the final result to ``out_features`` (skipped when
    ``in_features == out_features``).
    """

    glu_convs: list[eqx.nn.Conv1d]
    out_linear: eqx.nn.Linear | None
    in_features: int
    out_features: int

    def __init__(self, in_features: int, cfg: ConfigType, key: PRNGKeyArray):
        """Initialize the head."""
        dilations = cfg.dilation if cfg.dilation is not None else (1,) * cfg.num_layers
        if len(dilations) != cfg.num_layers:
            raise ValueError(
                f"dilation length {len(dilations)} != num_layers {cfg.num_layers}"
            )

        keys = jr.split(key, cfg.num_layers + 1)
        self.glu_convs = [
            eqx.nn.Conv1d(
                in_channels=1,
                out_channels=2,
                kernel_size=cfg.kernel_size,
                padding=_padding(cfg.kernel_size, d, cfg.causal),
                dilation=d,
                use_bias=cfg.use_bias,
                key=keys[i],
            )
            for i, d in enumerate(dilations)
        ]

        if in_features == cfg.out_features:
            self.out_linear = None
        else:
            self.out_linear = eqx.nn.Linear(
                in_features=in_features,
                out_features=cfg.out_features,
                use_bias=cfg.use_bias,
                key=keys[-1],
            )
        self.in_features = in_features
        self.out_features = cfg.out_features

    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        """Forward pass.

        Args:
            x: Input of shape ``(timesteps, in_features)``.
            state: Stateful layer state (passed through unchanged).

        Returns:
            Output of shape ``(timesteps, out_features)`` and unchanged state.
        """

        def per_frame(frame: Array) -> Array:
            h = frame[None, :]
            for conv in self.glu_convs:
                vg = conv(h)
                h = (vg[0] * jax.nn.sigmoid(vg[1]))[None, :]
            h = h.squeeze(0)
            if self.out_linear is not None:
                h = self.out_linear(h)
            return h

        return jax.vmap(per_frame)(x), state
