import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from linax.blocks import StandardBlockConfig, TFBlockConfig
from linax.encoder import ConvEncoderConfig, DenseEncoderConfig, LinearEncoderConfig
from linax.heads import (
    FCHeadConfig,
    MagDecoderHeadConfig,
    PhaseDecoderHeadConfig,
    RegressionHeadConfig,
)
from linax.models.linoss import LinOSSConfig
from linax.models.se_linoss import SELinOSSConfig
from linax.sequence_mixers import (
    LinOSSSequenceMixerConfig,
    MambaStyleLinOSSSequenceMixerConfig,
)
from linax.wrappers import NoiseCancellationWrapper, SpectralWrapper

__all__ = [
    "build_linoss",
    "build_linoss_spectral",
    "build_linoss_noise_cancellation",
    "build_se_linoss",
]


def build_linoss(subkey: PRNGKeyArray) -> eqx.Module:
    """Load a time-domain LinOSS model with given configuration."""
    hidden_size = 64
    cfg = LinOSSConfig(
        num_blocks=4,
        encoder_config=LinearEncoderConfig(in_features=1, out_features=hidden_size),
        sequence_mixer_config=LinOSSSequenceMixerConfig(
            state_dim=hidden_size, discretization="IMEX", damping=True, r_min=0.9, theta_max=jnp.pi
        ),
        block_config=StandardBlockConfig(drop_rate=0.1, prenorm=True),
        head_config=RegressionHeadConfig(out_features=1, reduce=False, normalize=True),
    )
    return cfg.build(key=subkey)


def build_linoss_receptive_field(subkey: PRNGKeyArray) -> eqx.Module:
    """Load a time-domain LinOSS model with given configuration.

    The encoder is a causal dilated 1D conv stack producing one u(t) per input
    sample with a ~5 ms receptive field. At 16 kHz, kernel=3 with dilations
    (1, 3, 9, 27) gives RF = 1 + 2*(1+3+9+27) = 81 samples ≈ 5.06 ms.
    """
    hidden_size = 64

    in_dims = (1, 16, 32, 32)
    out_dims = (16, 32, 32, 64)
    kernel_sizes = (3, 3, 3, 3)
    dilations = (1, 3, 9, 27)

    cfg = LinOSSConfig(
        num_blocks=4,
        encoder_config=ConvEncoderConfig(
            in_features=in_dims[0],
            out_features=out_dims[-1],
            in_dims=in_dims,
            out_dims=out_dims,
            kernel_size=kernel_sizes,
            dilation=dilations,
            causal=True,
        ),
        sequence_mixer_config=LinOSSSequenceMixerConfig(
            state_dim=hidden_size, discretization="IMEX", damping=True, r_min=0.9, theta_max=jnp.pi
        ),
        block_config=StandardBlockConfig(drop_rate=0.1, prenorm=True),
        head_config=RegressionHeadConfig(out_features=1, reduce=False, normalize=True),
    )
    return cfg.build(key=subkey)


def build_linoss_noise_cancellation(subkey: PRNGKeyArray) -> eqx.Module:
    """Load a time-domain LinOSS model wrapped for noise cancellation with given configuration."""
    hidden_size = 64
    cfg = LinOSSConfig(
        num_blocks=4,
        encoder_config=LinearEncoderConfig(in_features=1, out_features=hidden_size),
        sequence_mixer_config=LinOSSSequenceMixerConfig(
            state_dim=hidden_size, discretization="IMEX", damping=True, r_min=0.9, theta_max=jnp.pi
        ),
        block_config=StandardBlockConfig(drop_rate=0.1, prenorm=True),
        head_config=RegressionHeadConfig(out_features=1, reduce=False, normalize=True),
    )
    return NoiseCancellationWrapper(backbone=cfg.build(key=subkey))


def build_linoss_spectral(subkey: PRNGKeyArray) -> eqx.Module:
    """Load a spectral-domain LinOSS model with given configuration."""
    hidden_size = 64
    n_fft = 400
    n_bins = n_fft // 2 + 1

    cfg = LinOSSConfig(
        num_blocks=4,
        encoder_config=DenseEncoderConfig(in_channels=2, out_channels=hidden_size, dense_layers=4),
        sequence_mixer_config=LinOSSSequenceMixerConfig(
            state_dim=hidden_size,
            discretization="IMEX",
            damping=True,
            r_min=0.9,
            theta_max=jnp.pi,
        ),
        block_config=StandardBlockConfig(drop_rate=0.1, prenorm=True),
        head_config=FCHeadConfig(out_features=2 * n_bins, apply_activation=False),
    )

    return SpectralWrapper(
        generator=cfg.build(key=subkey),
        n_fft=n_fft,
        hop_length=100,
        win_length=400,
    )


def build_se_linoss(subkey: PRNGKeyArray) -> eqx.Module:
    """Load an SE-LinOSS (SEMamba-style time/freq LinOSS) model.

    Dense encoder → stack of TF blocks (separate time and freq LinOSS scans
    per block) → mag/phase decoders that upsample the freq axis and collapse
    channels. Wrapped in a ``SpectralWrapper`` for end-to-end waveform IO.
    """
    n_fft = 400
    n_bins = n_fft // 2 + 1

    cfg = SELinOSSConfig(
        num_blocks=2,
        encoder_config=DenseEncoderConfig(
            in_channels=2,
            out_channels=32,
            dense_layers=2,
            skip_type="residual",
        ),
        sequence_mixer_config=MambaStyleLinOSSSequenceMixerConfig(
            state_dim=32,
            expand=2,
            d_conv=4,
            causal_conv=True,
            discretization="IMEX",
            damping=True,
            r_min=0.9,
            theta_max=jnp.pi,
        ),
        block_config=TFBlockConfig(),
        mag_decoder_config=MagDecoderHeadConfig(
            dense_layers=1,
            dense_skip_type="residual",
            target_freq=n_bins,
            upsample_type="transposed",
            beta=2,
        ),
        phase_decoder_config=PhaseDecoderHeadConfig(
            dense_layers=1,
            dense_skip_type="residual",
            target_freq=n_bins,
            upsample_type="transposed",
        ),
    )

    return SpectralWrapper(
        generator=cfg.build(key=subkey),
        n_fft=n_fft,
        hop_length=100,  # 16 ms
        win_length=400,  # 32 ms
    )
