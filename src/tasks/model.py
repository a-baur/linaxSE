import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from linax.blocks import StandardBlockConfig
from linax.encoder import ConvEncoderConfig, LinearEncoderConfig
from linax.heads import RegressionHeadConfig
from linax.heads.fc import FCHeadConfig
from linax.models.linoss import LinOSSConfig
from linax.sequence_mixers import LinOSSSequenceMixerConfig
from linax.wrappers import NoiseCancellationWrapper, SpectralWrapper

__all__ = ["build_linoss", "build_linoss_spectral", "build_linoss_noise_cancellation"]


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
    hidden_size = 128
    n_fft = 512
    n_bins = n_fft // 2 + 1
    in_features = n_bins  # magnitude per bin

    in_dims = (in_features, 512, 256, 128)
    out_dims = (512, 256, 128, 256)
    kernel_sizes = (3, 3, 3, 3)

    cfg = LinOSSConfig(
        num_blocks=4,
        encoder_config=ConvEncoderConfig(
            in_features=in_dims[0],
            out_features=out_dims[-1],
            in_dims=in_dims,
            out_dims=out_dims,
            kernel_size=kernel_sizes,
        ),
        sequence_mixer_config=LinOSSSequenceMixerConfig(
            state_dim=hidden_size, discretization="IMEX", damping=True, r_min=0.9, theta_max=jnp.pi
        ),
        block_config=StandardBlockConfig(drop_rate=0.1, prenorm=True),
        head_config=FCHeadConfig(out_features=n_bins, apply_activation=False),
    )

    return SpectralWrapper(
        backbone=cfg.build(key=subkey),
        n_fft=n_fft,
        hop_length=256,  # 16 ms
        win_length=512,  # 32 ms
    )
