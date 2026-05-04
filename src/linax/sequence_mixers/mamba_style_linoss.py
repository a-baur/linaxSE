"""LinOSS wrapped in a Mamba-block-shaped surround.

The standard Mamba block does:

    in_proj(d_model -> 2*expand*d_model)  -> split into (u, z)
    causal depthwise Conv1d on u (kernel=d_conv, groups=expand*d_model)
    silu
    per-channel selective SSM on u (d_state per channel)
    u * silu(z)
    out_proj(expand*d_model -> d_model)

This module mirrors that structure but replaces the *selective* SSM with
LinOSS — a non-selective SSM. Each of the ``expand * d_model`` inner
channels owns its own LinOSS instance with ``in_features=1`` and
``state_dim=cfg.state_dim``; the channel-axis ensemble is constructed via
``eqx.filter_vmap`` and applied with another ``filter_vmap`` at call time.

LinOSS loses Mamba's input-dependent A/B/C selectivity — that's an
irreducible difference. Everything else (in_proj/out_proj, depthwise causal
conv, SiLU gating, per-channel SSM) is structurally faithful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from linax.sequence_mixers.base import SequenceMixer, SequenceMixerConfig
from linax.sequence_mixers.linoss import (
    LinOSSSequenceMixer,
    LinOSSSequenceMixerConfig,
)


@dataclass(frozen=True)
class MambaStyleLinOSSSequenceMixerConfig(SequenceMixerConfig):
    """Mamba-block-shaped wrapper around LinOSS.

    Attributes:
        state_dim: Per-channel SSM state size (Mamba's ``d_state``). Default 16.
        expand: Inner-channel expansion factor (Mamba's ``expand``). The inner
            dim used by the conv, gate, and per-channel SSMs is
            ``expand * in_features``. Default 4.
        d_conv: Depthwise conv kernel size along the sequence axis. Default 4.
        causal_conv: If True (Mamba-faithful), the depthwise conv is causal
            via left-padding so position ``t`` only sees positions ``≤ t``.
            Default True.
        discretization, damping, r_min, theta_max: Forwarded to the inner
            ``LinOSSSequenceMixerConfig`` for each per-channel SSM.
    """

    state_dim: int = 16
    expand: int = 4
    d_conv: int = 4
    causal_conv: bool = True

    # Forwarded to the inner per-channel LinOSS instances.
    discretization: Literal["IM", "IMEX"] = "IMEX"
    damping: bool = True
    r_min: float = 0.9
    theta_max: float = jnp.pi

    def build(
        self, in_features: int, key: PRNGKeyArray
    ) -> MambaStyleLinOSSSequenceMixer:
        """Build the Mamba-style LinOSS sequence mixer."""
        return MambaStyleLinOSSSequenceMixer(
            in_features=in_features, cfg=self, key=key
        )


class MambaStyleLinOSSSequenceMixer(SequenceMixer):
    """LinOSS in the Mamba block surround. Operates on ``(T, d_model)``.

    The inner per-channel SSMs are one ``LinOSSSequenceMixer`` module whose
    array leaves carry a leading axis of size ``inner_dim = expand * d_model``
    (built via ``eqx.filter_vmap`` so each channel gets independent params).
    """

    in_proj: eqx.nn.Linear
    conv: eqx.nn.Conv1d
    ssms: LinOSSSequenceMixer
    out_proj: eqx.nn.Linear

    inner_dim: int = eqx.field(static=True)
    d_conv: int = eqx.field(static=True)
    causal_conv: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        cfg: MambaStyleLinOSSSequenceMixerConfig,
        key: PRNGKeyArray,
    ):
        in_key, conv_key, ssm_key, out_key = jr.split(key, 4)

        inner_dim = cfg.expand * in_features
        self.inner_dim = inner_dim
        self.d_conv = cfg.d_conv
        self.causal_conv = cfg.causal_conv

        # in_proj produces (u, z) concatenated → split inside __call__.
        self.in_proj = eqx.nn.Linear(in_features, 2 * inner_dim, key=in_key)

        # Depthwise conv along the sequence axis. groups=inner_dim makes each
        # channel convolved with its own d_conv-tap kernel. Padding is handled
        # explicitly in __call__ so causal vs same-padding stays obvious.
        self.conv = eqx.nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=cfg.d_conv,
            groups=inner_dim,
            padding=0,
            key=conv_key,
        )

        # One tiny LinOSS per inner channel: in_features=1, state_dim=cfg.state_dim.
        # filter_vmap over keys produces a single module pytree with leading
        # axis `inner_dim` on every array leaf.
        inner_cfg = LinOSSSequenceMixerConfig(
            state_dim=cfg.state_dim,
            discretization=cfg.discretization,
            damping=cfg.damping,
            r_min=cfg.r_min,
            theta_max=cfg.theta_max,
        )
        ssm_keys = jr.split(ssm_key, inner_dim)
        make_ssm = eqx.filter_vmap(
            lambda k: LinOSSSequenceMixer(in_features=1, cfg=inner_cfg, key=k)
        )
        self.ssms = make_ssm(ssm_keys)

        self.out_proj = eqx.nn.Linear(inner_dim, in_features, key=out_key)

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward. ``x`` is ``(T, in_features)``; returns ``(T, in_features)``."""
        # in_proj per token: (T, in_features) → (T, 2*inner_dim) → split
        uz = jax.vmap(self.in_proj)(x)
        u, z = jnp.split(uz, 2, axis=-1)  # each (T, inner_dim)

        # Depthwise conv expects (channels, length). Pad before conv so output
        # length matches T regardless of causal/non-causal choice.
        u_ch = jnp.transpose(u, (1, 0))  # (inner_dim, T)
        if self.causal_conv:
            pad_left, pad_right = self.d_conv - 1, 0
        else:
            # "same" padding: split d_conv-1 across both sides (right gets the
            # extra when d_conv is even, matching common conv conventions).
            pad_left = (self.d_conv - 1) // 2
            pad_right = self.d_conv - 1 - pad_left
        u_ch = jnp.pad(u_ch, ((0, 0), (pad_left, pad_right)))
        u_ch = self.conv(u_ch)  # (inner_dim, T)
        u = jnp.transpose(u_ch, (1, 0))  # (T, inner_dim)
        u = jax.nn.silu(u)

        # Per-channel LinOSS scan. Ensemble the (T, 1) inputs, params, and
        # keys along the inner_dim axis and filter_vmap once.
        u_per_ch = jnp.transpose(u, (1, 0))[..., None]  # (inner_dim, T, 1)
        ssm_keys = jr.split(key, self.inner_dim)
        out_per_ch = eqx.filter_vmap(lambda ssm, xi, ki: ssm(xi, ki))(
            self.ssms, u_per_ch, ssm_keys
        )  # (inner_dim, T, 1)
        out = jnp.transpose(out_per_ch.squeeze(-1), (1, 0))  # (T, inner_dim)

        # Mamba-style multiplicative gating with SiLU(z).
        out = out * jax.nn.silu(z)

        # out_proj per token back to in_features.
        return jax.vmap(self.out_proj)(out)
