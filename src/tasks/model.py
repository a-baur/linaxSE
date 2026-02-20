import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from linax.blocks import StandardBlockConfig
from linax.encoder import LinearEncoderConfig
from linax.models.linoss import LinOSSConfig
from linax.heads import RegressionHeadConfig
from linax.sequence_mixers import LinOSSSequenceMixerConfig
from linax.wrappers import SpectralWrapper, NoiseCancellationWrapper


__all__ = ["build_linoss", "build_linoss_spectral", "build_linoss_noise_cancellation"]


def build_linoss(subkey: PRNGKeyArray) -> eqx.Module:
    """Load a time-domain LinOSS model with given configuration."""
    hidden_size = 64
    cfg = LinOSSConfig(
        num_blocks=4,
        encoder_config=LinearEncoderConfig(in_features=1, out_features=hidden_size),
        sequence_mixer_config=LinOSSSequenceMixerConfig(
            state_dim=hidden_size,
            discretization="IMEX",
            damping=True,
            r_min=0.9,
            theta_max=jnp.pi
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
            state_dim=hidden_size,
            discretization="IMEX",
            damping=True,
            r_min=0.9,
            theta_max=jnp.pi
        ),
        block_config=StandardBlockConfig(drop_rate=0.1, prenorm=True),
        head_config=RegressionHeadConfig(out_features=1, reduce=False, normalize=True),
    )
    return NoiseCancellationWrapper(backbone=cfg.build(key=subkey))


def build_linoss_spectral(subkey: PRNGKeyArray) -> eqx.Module:
    """Load a spectral-domain LinOSS model with given configuration."""
    hidden_size = 64
    n_fft = 512

    in_features = (n_fft // 2 + 1)  # Real and imaginary parts of the spectrogram
    cfg = LinOSSConfig(
        num_blocks=4,
        encoder_config=LinearEncoderConfig(in_features=in_features, out_features=hidden_size),
        sequence_mixer_config=LinOSSSequenceMixerConfig(
            state_dim=hidden_size,
            discretization="IMEX",
            damping=True,
            r_min=0.9,
            theta_max=jnp.pi
        ),
        block_config=StandardBlockConfig(drop_rate=0.1, prenorm=True),
        head_config=RegressionHeadConfig(out_features=in_features, reduce=False, normalize=False),
    )
    return SpectralWrapper(
        backbone=cfg.build(key=subkey),
        n_fft=n_fft,
        hop_length=256,
        win_length=512,
        power=0.3
    )
