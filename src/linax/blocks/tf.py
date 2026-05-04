"""Time-Frequency block. Two LinOSS sequence mixers (time + freq), à la SEMamba."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from linax.channel_mixers.base import ChannelMixer
from linax.sequence_mixers.base import SequenceMixer


@dataclass(frozen=True)
class TFBlockConfig:
    """Configuration for a Time-Frequency block.

    Attributes:
        drop_rate: Dropout rate applied after the sequence and channel mixers.
        prenorm: Whether to apply normalisation before (True) or after (False) the
            residual.
    """

    drop_rate: float = 0.1
    prenorm: bool = True

    def build(
            self,
            in_features: int,
            time_sequence_mixer: SequenceMixer,
            freq_sequence_mixer: SequenceMixer,
            time_channel_mixer: ChannelMixer,
            freq_channel_mixer: ChannelMixer,
            key: PRNGKeyArray,
    ) -> TFBlock:
        """Build a TFBlock from this config.

        Unlike `StandardBlockConfig.build`, the caller must supply *two* sequence
        mixers and *two* channel mixers — one per direction. They are kept as
        independent sub-modules so their parameters and BatchNorm running stats
        do not mix.
        """
        return TFBlock(
            in_features=in_features,
            cfg=self,
            time_sequence_mixer=time_sequence_mixer,
            freq_sequence_mixer=freq_sequence_mixer,
            time_channel_mixer=time_channel_mixer,
            freq_channel_mixer=freq_channel_mixer,
            key=key,
        )


class TFBlock(eqx.Module):
    """Time-Frequency block with two independent sequence-mixer scans.

    The block runs two residual sub-blocks back-to-back on a ``[C, T, F]`` tensor:
    a *time* scan along the frame axis (vmapped over bins) and a *freq* scan along
    the bin axis (vmapped over frames). Each direction owns its sequence mixer,
    channel mixer, and BatchNorm — they are separate ``eqx.Module`` leaves, so
    their parameters and BN state stay disjoint.

    LayerNorm is applied per (frame, bin) token over the channel axis, matching
    SEMamba's TF-Mamba block. The norm lives inside each branch and is invoked
    after the transpose so the channel axis is last, removing the previous
    BatchNorm dependency on ``axis_name="batch"`` and on a vmapped state.
    """

    time_norm: eqx.nn.LayerNorm
    time_sequence_mixer: SequenceMixer
    time_channel_mixer: ChannelMixer

    freq_norm: eqx.nn.LayerNorm
    freq_sequence_mixer: SequenceMixer
    freq_channel_mixer: ChannelMixer

    drop: eqx.nn.Dropout
    prenorm: bool

    def __init__(
            self,
            in_features: int,
            cfg: TFBlockConfig,
            time_sequence_mixer: SequenceMixer,
            freq_sequence_mixer: SequenceMixer,
            time_channel_mixer: ChannelMixer,
            freq_channel_mixer: ChannelMixer,
            key: PRNGKeyArray,
    ):
        self.time_sequence_mixer = time_sequence_mixer
        self.freq_sequence_mixer = freq_sequence_mixer
        self.time_channel_mixer = time_channel_mixer
        self.freq_channel_mixer = freq_channel_mixer

        self.time_norm = eqx.nn.LayerNorm(shape=in_features)
        self.freq_norm = eqx.nn.LayerNorm(shape=in_features)
        self.drop = eqx.nn.Dropout(p=cfg.drop_rate)
        self.prenorm = cfg.prenorm

    def __call__(
            self,
            x: Float[Array, "channels frames bins"],
            state: eqx.nn.State,
            key: PRNGKeyArray,
    ) -> tuple[Float[Array, "channels frames bins"], eqx.nn.State]:
        time_key, freq_key = jr.split(key)

        # Time scan: [T, C] over F
        x = self._branch(
            x,
            norm=self.time_norm,
            seq=self.time_sequence_mixer,
            chan=self.time_channel_mixer,
            forward_perm=(2, 1, 0),  # [C, T, F] -> [F, T, C]
            inverse_perm=(2, 1, 0),  # [F, T, C] -> [C, T, F]
            key=time_key,
        )
        # Freq scan: [F, C] over T
        x = self._branch(
            x,
            norm=self.freq_norm,
            seq=self.freq_sequence_mixer,
            chan=self.freq_channel_mixer,
            forward_perm=(1, 2, 0),  # [C, T, F] -> [T, F, C]
            inverse_perm=(2, 0, 1),  # [T, F, C] -> [C, T, F]
            key=freq_key,
        )
        return x, state

    def _branch(
            self,
            x: Float[Array, "channels frames bins"],
            norm: eqx.nn.LayerNorm,
            seq: SequenceMixer,
            chan: ChannelMixer,
            forward_perm: tuple[int, int, int],
            inverse_perm: tuple[int, int, int],
            key: PRNGKeyArray,
    ) -> Float[Array, "channels frames bins"]:
        seq_key, drop_key1, drop_key2 = jr.split(key, 3)

        skip = x
        h = jnp.transpose(x, forward_perm)  # channel axis is last
        if self.prenorm:
            h = jax.vmap(jax.vmap(norm))(h)
        h = jax.vmap(lambda hi: seq(hi, seq_key))(h)
        h = self.drop(jax.nn.gelu(h), key=drop_key1)
        h = jax.vmap(jax.vmap(chan))(h)
        h = self.drop(h, key=drop_key2)
        h = jnp.transpose(h, inverse_perm)

        x = skip + h
        if not self.prenorm:
            h_ln = jnp.moveaxis(x, 0, -1)
            h_ln = jax.vmap(jax.vmap(norm))(h_ln)
            x = jnp.moveaxis(h_ln, -1, 0)
        return x
