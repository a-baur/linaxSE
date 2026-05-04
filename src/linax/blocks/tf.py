"""Time-Frequency block. SEMamba-style ``TFMambaBlock`` wrapper around a LinOSS SSM."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

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
        time_sequence_mixer: SequenceMixer,
        freq_sequence_mixer: SequenceMixer,
        key: PRNGKeyArray,
    ) -> TFBlock:
        """Build a TFBlock from this config.

        The caller supplies *two* sequence mixers — one per direction. They
        are kept as independent sub-modules so their parameters do not mix.
        """
        return TFBlock(
            in_features=in_features,
            time_sequence_mixer=time_sequence_mixer,
            freq_sequence_mixer=freq_sequence_mixer,
            key=key,
        )


class TFBlock(eqx.Module):
    """Time-Frequency block mirroring SEMamba's ``TFMambaBlock`` (unidirectional).

    For each branch (time, then freq) the wrapper does:

    1. transpose so the channel axis is last;
    2. ``LayerNorm`` over channels (per (F, T) token) — supplies the pre-norm
       that mamba-ssm's ``Block`` provides for free in SEMamba;
    3. apply the sequence mixer along the seq axis (vmapped over the other);
    4. add back the pre-norm input — the inner residual that ``Block``'s
       residual stream produces in the SEMamba reference;
    5. apply a per-token ``Linear(C → C)`` — the analog of SEMamba's
       ``tlinear``/``flinear`` (``ConvTranspose1d(2C → C, 1, 1)``); with the
       unidirectional scan there's no ``2C`` to collapse, so the projection
       degenerates to a learnable C→C mix;
    6. transpose back and add the outer residual.

    Compared to SEMamba's ``TFMambaBlock`` the only structural difference is
    the unidirectional scan: SEMamba runs the SSM forward and on the flipped
    sequence, then concatenates the outputs along channels (giving ``2C``)
    before the linear projection. Here the SSM runs once, and the projection
    is ``C → C``.
    """

    time_norm: eqx.nn.LayerNorm
    time_sequence_mixer: SequenceMixer
    time_proj: eqx.nn.Linear

    freq_norm: eqx.nn.LayerNorm
    freq_sequence_mixer: SequenceMixer
    freq_proj: eqx.nn.Linear

    def __init__(
        self,
        in_features: int,
        time_sequence_mixer: SequenceMixer,
        freq_sequence_mixer: SequenceMixer,
        key: PRNGKeyArray,
    ):
        tp_key, fp_key = jr.split(key, 2)

        self.time_sequence_mixer = time_sequence_mixer
        self.freq_sequence_mixer = freq_sequence_mixer

        self.time_norm = eqx.nn.LayerNorm(shape=in_features)
        self.freq_norm = eqx.nn.LayerNorm(shape=in_features)

        self.time_proj = eqx.nn.Linear(in_features, in_features, key=tp_key)
        self.freq_proj = eqx.nn.Linear(in_features, in_features, key=fp_key)

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
            norm=self.time_norm,
            seq=self.time_sequence_mixer,
            proj=self.time_proj,
            forward_perm=(2, 1, 0),  # [C, T, F] -> [F, T, C]
            inverse_perm=(2, 1, 0),  # [F, T, C] -> [C, T, F]
            key=time_key,
        )
        # Freq scan: per (T,) → mixer along F, channels last
        x = self._branch(
            x,
            norm=self.freq_norm,
            seq=self.freq_sequence_mixer,
            proj=self.freq_proj,
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
        proj: eqx.nn.Linear,
        forward_perm: tuple[int, int, int],
        inverse_perm: tuple[int, int, int],
        key: PRNGKeyArray,
    ) -> Float[Array, "channels frames bins"]:
        skip_outer = x
        h_in = jnp.transpose(x, forward_perm)
        h = jax.vmap(jax.vmap(norm))(h_in)
        h = jax.vmap(lambda hi: seq(hi, key))(h)
        h = h + h_in
        h = jax.vmap(jax.vmap(proj))(h)
        h = jnp.transpose(h, inverse_perm)
        return skip_outer + h
