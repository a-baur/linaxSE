"""SE-LinOSS — SEMamba-style time/frequency LinOSS for speech enhancement."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from linax.blocks.tf import TFBlock, TFBlockConfig
from linax.channel_mixers.glu import GLUConfig
from linax.encoder.base import Encoder, EncoderConfig
from linax.heads.base import Head, HeadConfig
from linax.sequence_mixers.linoss import LinOSSSequenceMixerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SELinOSSConfig:
    """Configuration for SE-LinOSS.

    The model encodes the noisy STFT into ``[C, T, F/2]``, processes it through
    ``num_blocks`` TF blocks (each with separate time and frequency LinOSS scans
    that do not share parameters or state), then decodes the magnitude mask and
    phase via two independent heads.

    The shared ``sequence_mixer_config`` and ``channel_mixer_config`` define the
    *shape* of the time and frequency mixers; each TF block instantiates them
    twice with independent keys, so parameters are not tied across directions.

    Attributes:
        num_blocks: Number of stacked TF blocks.
        encoder_config: Encoder config (typically a ``DenseEncoderConfig``)
            producing ``[out_channels, frames, bins // 2]``.
        mag_decoder_config: Head producing the per-frame magnitude mask.
            Receives ``[C, T, F/2]`` directly; typically a
            ``MagDecoderHeadConfig``.
        phase_decoder_config: Head producing the per-frame phase prediction;
            typically a ``PhaseDecoderHeadConfig``.
        sequence_mixer_config: LinOSS config used for both time and freq mixers.
        block_config: TF block config (drop rate, prenorm).
        channel_mixer_config: GLU config used for both time and freq channel
            mixers.
    """

    num_blocks: int
    encoder_config: EncoderConfig
    mag_decoder_config: HeadConfig
    phase_decoder_config: HeadConfig
    sequence_mixer_config: LinOSSSequenceMixerConfig = field(
        default_factory=LinOSSSequenceMixerConfig
    )
    block_config: TFBlockConfig = field(default_factory=TFBlockConfig)
    channel_mixer_config: GLUConfig = field(default_factory=GLUConfig)

    def build(self, key: PRNGKeyArray | None = None) -> SELinOSS:
        """Build the SE-LinOSS model from this configuration."""
        if key is None:
            logger.warning("No key provided. Set automatically.")
            key = jr.PRNGKey(0)
        return SELinOSS(cfg=self, key=key)


class SELinOSS(eqx.Module):
    """Dense encoder + stack of TF blocks + dual mag/phase decoder heads.

    Forward pass on input ``[2, frames, bins]`` (compressed mag + phase from the
    spectral wrapper):

    1. ``encoder``: ``[2, T, F]`` → ``[C, T, F/2]``.
    2. ``blocks``: ``[C, T, F/2]`` → ``[C, T, F/2]`` (preserves shape).
    3. ``mag_decoder`` and ``phase_decoder`` each take ``[C, T, F/2]`` and
       return ``[T, F]`` (channels collapsed, freq axis upsampled).
    4. concatenate along the feature axis: ``[T, 2F]`` so the spectral wrapper
       can ``jnp.split(out, 2, axis=1)``.
    """

    encoder: Encoder
    blocks: list[TFBlock]
    mag_decoder: Head
    phase_decoder: Head

    def __init__(self, cfg: SELinOSSConfig, key: PRNGKeyArray):
        enc_key, blocks_key, mag_key, phase_key = jr.split(key, 4)

        self.encoder = cfg.encoder_config.build(key=enc_key)

        hidden_dim = cfg.encoder_config.out_channels
        block_keys = jr.split(blocks_key, cfg.num_blocks)
        self.blocks = [
            self._build_block(cfg, hidden_dim=hidden_dim, key=k) for k in block_keys
        ]

        self.mag_decoder = cfg.mag_decoder_config.build(
            in_features=hidden_dim, key=mag_key
        )
        self.phase_decoder = cfg.phase_decoder_config.build(
            in_features=hidden_dim, key=phase_key
        )

    @staticmethod
    def _build_block(
        cfg: SELinOSSConfig, hidden_dim: int, key: PRNGKeyArray
    ) -> TFBlock:
        ts_key, fs_key, tc_key, fc_key, b_key = jr.split(key, 5)
        return cfg.block_config.build(
            in_features=hidden_dim,
            time_sequence_mixer=cfg.sequence_mixer_config.build(
                in_features=hidden_dim, key=ts_key
            ),
            freq_sequence_mixer=cfg.sequence_mixer_config.build(
                in_features=hidden_dim, key=fs_key
            ),
            time_channel_mixer=cfg.channel_mixer_config.build(
                in_features=hidden_dim, out_features=None, key=tc_key
            ),
            freq_channel_mixer=cfg.channel_mixer_config.build(
                in_features=hidden_dim, out_features=None, key=fc_key
            ),
            key=b_key,
        )

    def __call__(
        self,
        x: Float[Array, "stack frames bins"],
        state: eqx.nn.State,
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "frames feat"], eqx.nn.State]:
        block_keys = jr.split(key, len(self.blocks))

        x, state = self.encoder(x, state)  # [C, T, F/2]
        for block, k in zip(self.blocks, block_keys):
            x, state = block(x, state, k)

        mag_out, state = self.mag_decoder(x, state)  # [T, F]
        phase_out, state = self.phase_decoder(x, state)  # [T, F]
        return jnp.concatenate([mag_out, phase_out], axis=1), state
