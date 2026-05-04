import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, PRNGKeyArray

from linax.models import SSM
from linax.wrappers.output import ModelOutput


class SpectralWrapper(eqx.Module):
    """A wrapper that applies a SSM backbone in the spectral domain.

    Args:
        generator: The SSM module to generate the magnitude.
        n_fft: Number of FFT points for STFT.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
    """

    generator: SSM
    inference: bool
    n_fft: int = eqx.field(static=True)
    hop_length: int = eqx.field(static=True)
    win_length: int = eqx.field(static=True)

    def __init__(
        self,
        generator: SSM,
        n_fft: int = 400,
        hop_length: int = 100,
        win_length: int = 400,
        inference: bool = False,
    ):
        self.generator = generator
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.inference = inference

    def __call__(
        self, x: Float[jax.Array, "time bins"], state: eqx.nn.State, key: PRNGKeyArray
    ) -> tuple[ModelOutput, eqx.nn.State]:
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
        mag_c = mag**p
        in_features = jnp.stack([mag_c, phase], axis=0)  # [2, frames, bins]

        out, new_state = self.generator(in_features, state, key)

        mag_mask_out, phase_out = jnp.split(out, 2, axis=1)

        mag_mask_pred = mag_mask_out.reshape(n_frames, n_bins)
        mag_c_pred = jnp.abs(mag_c * mag_mask_pred)
        phase_pred = phase_out.reshape(n_frames, n_bins)

        # Decompress predicted magnitude and recombine with the noisy phase.
        mag_pred = mag_c_pred ** (1.0 / p)

        Zxx_enhanced = (mag_pred * jnp.exp(1j * phase_pred)).T  # [bins, frames]
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

        output = ModelOutput(
            prediction=x_recon,
            aux={"mag_c_pred": mag_c_pred, "mag_pred": mag_pred, "phase_pred": phase_pred},
        )
        return output, new_state
