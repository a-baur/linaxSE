import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from linax.models import SSM


class SpectralWrapper(eqx.Module):
    """ A wrapper that applies a SSM backbone in the spectral domain.

    Args:
        backbone: The SSM module to apply in the spectral domain.
        n_fft: Number of FFT points for STFT.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
    """
    backbone: SSM
    n_fft: int = eqx.field(static=True)
    hop_length: int = eqx.field(static=True)
    win_length: int = eqx.field(static=True)

    def __init__(
            self,
            backbone: SSM,
            n_fft: int = 512,
            hop_length: int = 256,
            win_length: int = 512
    ):
        self.backbone = backbone
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def __call__(
            self,
            x: Array,
            state: eqx.nn.State,
            key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        """
        Args:
            x: Input waveform [Time]
            state: Backbone state
            key: Random key
        """
        original_length = x.shape[0]

        _, _, Zxx = jax.scipy.signal.stft(
            x.squeeze(),
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
        )

        # [freq, frames] -> [frames, freq]
        Zxx = Zxx.T
        mag = jnp.abs(Zxx)
        phase = jnp.angle(Zxx)

        backbone_x = jnp.concatenate([mag, phase], axis=-1)

        processed_spec, new_state = self.backbone(backbone_x, state, key)

        n_bins = Zxx.shape[-1]
        mag = processed_spec[..., :n_bins]
        phase = processed_spec[..., n_bins:]
        Zxx_out = mag * jnp.exp(1j * phase)

        # [frames, freq] -> [freq, frames]
        Zxx_out = Zxx_out.T

        _, x_recon = jax.scipy.signal.istft(
            Zxx_out,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft
        )

        # adjust length to match input (due to padding in STFT)
        if x_recon.shape[0] > original_length:
            x_recon = x_recon[:original_length]
        elif x_recon.shape[0] < original_length:
            diff = original_length - x_recon.shape[0]
            x_recon = jnp.pad(x_recon, ((0, diff),))
        x_recon = x_recon[:, None]  # reintroduce last dimension

        return x_recon, new_state
