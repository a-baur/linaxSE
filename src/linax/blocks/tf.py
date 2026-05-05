"""Time-Frequency block. SEMamba-style ``TFMambaBlock`` wrapper around a LinOSS SSM."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from linax.modules import RMSNorm
from linax.sequence_mixers.base import SequenceMixer


@dataclass(frozen=True)
class TFBlockConfig:
    """Configuration for a Time-Frequency block.

    The block has no tunable knobs at the wrapper level: norm placement is
    fixed (pre-mixer LayerNorm), there is no dropout, no activation, and no
    channel mixer — matching SEMamba's ``TFMambaBlock`` structure. The only
    way to vary the block is to swap in different sequence mixers.
    """

    def build(
        self,
        in_features: int,
        time_sequence_mixer_fwd: SequenceMixer,
        time_sequence_mixer_bwd: SequenceMixer,
        freq_sequence_mixer_fwd: SequenceMixer,
        freq_sequence_mixer_bwd: SequenceMixer,
        key: PRNGKeyArray,
    ) -> TFBlock:
        """Build a TFBlock from this config.

        The caller supplies *four* sequence mixers — one per (branch, scan
        direction). They are kept as independent sub-modules so their
        parameters do not mix.
        """
        return TFBlock(
            in_features=in_features,
            time_sequence_mixer_fwd=time_sequence_mixer_fwd,
            time_sequence_mixer_bwd=time_sequence_mixer_bwd,
            freq_sequence_mixer_fwd=freq_sequence_mixer_fwd,
            freq_sequence_mixer_bwd=freq_sequence_mixer_bwd,
            key=key,
        )


class TFBlock(eqx.Module):
    """Time-Frequency block mirroring SEMamba's ``TFMambaBlock`` (bidirectional).

    For each branch (time, then freq) the wrapper does:

    1. transpose so the channel axis is last;
    2. run the SSM in two independent directions:

       - forward: ``RMSNorm → SSM → +input`` (inner residual);
       - backward: flip the seq axis, ``RMSNorm → SSM → +flipped_input``
         (inner residual), then flip back.

       Each direction has its own ``RMSNorm`` and its own SSM; parameters
       are not shared (matches SEMamba's ``MambaBlock`` which holds two
       independent ``Block``\\s).

    3. concatenate the two direction outputs along the channel axis,
       giving ``2C``;
    4. apply a per-token ``Linear(2C → C)`` — the analog of SEMamba's
       ``tlinear``/``flinear`` (``ConvTranspose1d(2C, C, kernel=1, stride=1)``,
       which is structurally just a per-token affine);
    5. transpose back and add the outer residual.
    """

    time_norm_fwd: RMSNorm
    time_norm_bwd: RMSNorm
    time_sequence_mixer_fwd: SequenceMixer
    time_sequence_mixer_bwd: SequenceMixer
    time_proj: eqx.nn.Linear

    freq_norm_fwd: RMSNorm
    freq_norm_bwd: RMSNorm
    freq_sequence_mixer_fwd: SequenceMixer
    freq_sequence_mixer_bwd: SequenceMixer
    freq_proj: eqx.nn.Linear

    def __init__(
        self,
        in_features: int,
        time_sequence_mixer_fwd: SequenceMixer,
        time_sequence_mixer_bwd: SequenceMixer,
        freq_sequence_mixer_fwd: SequenceMixer,
        freq_sequence_mixer_bwd: SequenceMixer,
        key: PRNGKeyArray,
    ):
        tp_key, fp_key = jr.split(key, 2)

        self.time_sequence_mixer_fwd = time_sequence_mixer_fwd
        self.time_sequence_mixer_bwd = time_sequence_mixer_bwd
        self.freq_sequence_mixer_fwd = freq_sequence_mixer_fwd
        self.freq_sequence_mixer_bwd = freq_sequence_mixer_bwd

        self.time_norm_fwd = RMSNorm(in_features)
        self.time_norm_bwd = RMSNorm(in_features)
        self.freq_norm_fwd = RMSNorm(in_features)
        self.freq_norm_bwd = RMSNorm(in_features)

        self.time_proj = eqx.nn.Linear(2 * in_features, in_features, key=tp_key)
        self.freq_proj = eqx.nn.Linear(2 * in_features, in_features, key=fp_key)

    def __call__(
        self,
        x: Float[Array, "channels frames bins"],
        state: eqx.nn.State,
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "channels frames bins"], eqx.nn.State]:
        time_key, freq_key = jr.split(key)

        # Time scan: per (F,) → mixer along T, channels last
        x = self._branch(
            x,
            norm_fwd=self.time_norm_fwd,
            norm_bwd=self.time_norm_bwd,
            seq_fwd=self.time_sequence_mixer_fwd,
            seq_bwd=self.time_sequence_mixer_bwd,
            proj=self.time_proj,
            forward_perm=(2, 1, 0),  # [C, T, F] -> [F, T, C]
            inverse_perm=(2, 1, 0),  # [F, T, C] -> [C, T, F]
            key=time_key,
        )
        # Freq scan: per (T,) → mixer along F, channels last
        x = self._branch(
            x,
            norm_fwd=self.freq_norm_fwd,
            norm_bwd=self.freq_norm_bwd,
            seq_fwd=self.freq_sequence_mixer_fwd,
            seq_bwd=self.freq_sequence_mixer_bwd,
            proj=self.freq_proj,
            forward_perm=(1, 2, 0),  # [C, T, F] -> [T, F, C]
            inverse_perm=(2, 0, 1),  # [T, F, C] -> [C, T, F]
            key=freq_key,
        )
        return x, state

    def _branch(
        self,
        x: Float[Array, "channels frames bins"],
        norm_fwd: RMSNorm,
        norm_bwd: RMSNorm,
        seq_fwd: SequenceMixer,
        seq_bwd: SequenceMixer,
        proj: eqx.nn.Linear,
        forward_perm: tuple[int, int, int],
        inverse_perm: tuple[int, int, int],
        key: PRNGKeyArray,
    ) -> Float[Array, "channels frames bins"]:
        fwd_key, bwd_key = jr.split(key)
        skip_outer = x
        h_in = jnp.transpose(x, forward_perm)  # (other, seq, C)

        # Forward direction
        f = jax.vmap(jax.vmap(norm_fwd))(h_in)
        f = jax.vmap(lambda hi: seq_fwd(hi, fwd_key))(f)
        f = f + h_in  # inner residual

        # Backward direction (flip the seq axis only, axis 1 after permutation)
        b_in = jnp.flip(h_in, axis=1)
        b = jax.vmap(jax.vmap(norm_bwd))(b_in)
        b = jax.vmap(lambda hi: seq_bwd(hi, bwd_key))(b)
        b = b + b_in  # inner residual (vs. flipped input, then unflip)
        b = jnp.flip(b, axis=1)

        # Concat along channels (last) and project 2C → C per token
        h = jnp.concatenate([f, b], axis=-1)  # (other, seq, 2C)
        h = jax.vmap(jax.vmap(proj))(h)  # (other, seq, C)
        h = jnp.transpose(h, inverse_perm)
        return skip_outer + h
