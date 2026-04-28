import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, PRNGKeyArray

from linax.models import SSM


class SpectralWrapper(eqx.Module):
    """A wrapper that applies a SSM backbone in the spectral domain.

    Args:
        backbone: The SSM module to apply in the spectral domain.
        n_fft: Number of FFT points for STFT.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
    """

    backbone: SSM
    inference: bool
    n_fft: int = eqx.field(static=True)
    hop_length: int = eqx.field(static=True)
    win_length: int = eqx.field(static=True)

    def __init__(
            self,
            backbone: SSM,
            n_fft: int = 512,
            hop_length: int = 256,
            win_length: int = 512,
            inference: bool = False,
    ):
        self.backbone = backbone
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.inference = inference

    def __call__(
            self, x: Float[jax.Array, "time bins"], state: eqx.nn.State, key: PRNGKeyArray
    ) -> tuple[Float[jax.Array, "time bins"], eqx.nn.State]:
        """Forward pass of Spectral Wrapper."""
        original_length = x.shape[0]
        p = 0.3

        _, _, Zxx = jax.scipy.signal.stft(
            x.squeeze(),
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
        )

        Zxx = Zxx.T  # [frames, bins]
        n_frames, n_bins = Zxx.shape
        mag = jnp.abs(Zxx)
        phase = jnp.angle(Zxx)

        # Compressed magnitude as network input.
        mag_c = mag ** p

        # Magnitude-only mapping: backbone predicts compressed magnitude of the clean spectrum.
        out, new_state = self.backbone(mag_c, state, key)
        mag_c_pred = out.reshape(n_frames, n_bins)

        # Decompress predicted magnitude and recombine with the noisy phase.
        mag_pred = jnp.abs(mag_c_pred) ** (1.0 / p)

        Zxx_enhanced = (mag_pred * jnp.exp(1j * phase)).T  # [bins, frames]
        _, x_recon = jax.scipy.signal.istft(
            Zxx_enhanced,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
        )

        # adjust length to match input (due to padding in STFT)
        if x_recon.shape[0] > original_length:
            x_recon = x_recon[:original_length]
        elif x_recon.shape[0] < original_length:
            diff = original_length - x_recon.shape[0]
            x_recon = jnp.pad(x_recon, ((0, diff),))
        x_recon = x_recon[:, None]  # reintroduce last dimension

        return x_recon, new_state
