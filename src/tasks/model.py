import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from linax.blocks import StandardBlockConfig
from linax.encoder import LinearEncoderConfig
from linax.models import SSM
from linax.models.linoss import LinOSSConfig
from linax.heads import RegressionHeadConfig
from linax.sequence_mixers import LinOSSSequenceMixerConfig

__all__ = ["build_linoss_time"]


def build_linoss_time(subkey: PRNGKeyArray) -> SSM:
    """Load a time-domain LinOSS model with given configuration.

    Args:
        subkey (PRNGKeyArray): Random key to initialize the model.

    Returns:
        tuple[eqx.Module, eqx.nn.State]: The initialized LinOSS model and its state.
    """
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
    model = cfg.build(key=subkey)
    return model
