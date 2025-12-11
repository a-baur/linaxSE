import equinox as eqx
from jaxtyping import PRNGKeyArray

from linax.encoder import LinearEncoderConfig
from linax.models import SSM
from linax.models.linoss import LinOSSConfig
from linax.heads import RegressionHeadConfig

__all__ = ["build_linoss_model"]


def build_linoss_model(
    num_blocks: int, hidden_size: int, subkey: PRNGKeyArray
) -> tuple[SSM, eqx.nn.State]:
    """Load a LinOSS model with given configuration.

    Args:
        num_blocks (int): Number of LinOSS blocks.
        hidden_size (int): Hidden size of the encoder.
        subkey (PRNGKeyArray): Random key to initialize the model.

    Returns:
        tuple[eqx.Module, eqx.nn.State]: The initialized LinOSS model and its state.
    """
    cfg = LinOSSConfig(
        num_blocks=num_blocks,
        encoder_config=LinearEncoderConfig(in_features=1, out_features=hidden_size),
        head_config=RegressionHeadConfig(out_features=1, reduce=False),
    )
    model = cfg.build(key=subkey)
    state = eqx.nn.State(model=model)
    return model, state
