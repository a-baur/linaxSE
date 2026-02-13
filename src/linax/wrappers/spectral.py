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
    power: float = eqx.field(static=True)

    def __init__(
            self,
            backbone: SSM,
            n_fft: int = 512,
            hop_length: int = 256,
            win_length: int = 512,
            power: float = 0.3,
    ):
        self.backbone = backbone
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.power = power

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
        mag_comp = mag ** self.power
        Zxx_comp = mag_comp * jnp.exp(1j * phase)

        mask_logits, new_state = self.backbone(Zxx_comp.real, state, key)

        # Bound the mask (e.g., via tanh) to prevent output explosion
        mask_real = jnp.tanh(mask_logits)
        complex_mask = mask_real + 1j * Zxx_comp.imag

        # Apply predicted mask to the uncompressed input STFT
        Zxx_enhanced = Zxx * complex_mask

        # 5. ISTFT and Overlap-Add
        Zxx_enhanced = Zxx_enhanced.T  # [freq, frames]

        _, x_recon = jax.scipy.signal.istft(
            Zxx_enhanced,
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
