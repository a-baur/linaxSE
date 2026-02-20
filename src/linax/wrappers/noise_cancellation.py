import equinox as eqx
from jax import Array
from jaxtyping import PRNGKeyArray, Float

from linax.models import SSM


class NoiseCancellationWrapper(eqx.Module):
    """ A wrapper that applies a SSM backbone in the time domain for noise cancellation.

    Args:
        backbone: The SSM module to apply in the time domain for noise cancellation.
    """
    backbone: SSM

    def __init__(self, backbone: SSM):
        self.backbone = backbone

    def __call__(
            self,
            x: Float[Array, "time 1"],
            state: eqx.nn.State,
            key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        """
        Args:
            x: Input waveform [Time]
            state: Backbone state
            key: Random key
        """
        pred_n, state = self.backbone(x, state, key)
        pred_y = x - pred_n
        return pred_y, state
