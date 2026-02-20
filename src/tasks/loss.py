import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jax import Array
from jaxtyping import Float, Int
from pesq import pesq


@eqx.filter_jit
def l1_loss(
    y: Float[Array, "batch time feature"],
    pred_y: Float[Array, "batch time feature"],
    mask: Int[Array, "batch time feature"],
) -> Float[Array, ""]:
    l1 = jnp.sum(jnp.abs(pred_y - y) * mask) / jnp.sum(mask)
    return l1


@eqx.filter_jit
def mse_loss(
    y: Float[Array, "batch time feature"],
    pred_y: Float[Array, "batch time feature"],
    mask: Int[Array, "batch time feature"],
) -> Float[Array, ""]:
    mse = jnp.sum(((pred_y - y) ** 2) * mask) / jnp.sum(mask)
    return mse


@eqx.filter_jit
def si_sdr_loss(
    y: Float[Array, "batch time feature"],
    pred_y: Float[Array, "batch time feature"],
    mask: Int[Array, "batch time feature"],
    zero_mean: bool = True,
    invert: bool = True,
) -> Float[Array, ""]:
    """
    Computes Negative SI-SDR loss for a batch.

    Args:
        pred_y: (Batch, Time, 1) Estimated audio
        y: (Batch, Time, 1) Clean target audio
        mask: (Batch, Time, 1) Binary mask for valid lengths
        zero_mean: Whether to zero-mean the signals before computation
        invert (bool): Whether to return negative SI-SDR for minimization (default True)

    Returns:
        Negative SI-SDR loss
    """
    pred_y = pred_y.squeeze(-1) * mask.squeeze(-1)
    y = y.squeeze(-1) * mask.squeeze(-1)

    if zero_mean:
        pred_y = pred_y - jnp.mean(pred_y, axis=-1, keepdims=True)
        y = y - jnp.mean(y, axis=-1, keepdims=True)

    eps = jax.numpy.finfo(pred_y.dtype).eps

    alpha = (
        (jnp.sum(pred_y * y, axis=-1, keepdims=True) + eps)
        / (jnp.sum(y ** 2, axis=-1, keepdims=True) + eps)
    )
    target_scaled = alpha * y

    noise = target_scaled - pred_y

    target_pow = jnp.sum(target_scaled ** 2, axis=1) + eps
    noise_pow = jnp.sum(noise ** 2, axis=1) + eps
    si_sdr = 10.0 * jnp.log10(target_pow / noise_pow)

    return jnp.mean(si_sdr) * (-1 if invert else 1)


def pesq_loss(
    y: Float[Array, "batch time feature"],
    pred_y: Float[Array, "batch time feature"],
    mask: Int[Array, "batch time feature"],
) -> Float[Array, ""]:
    loss = 0
    for i in range(y.shape[0]):
        end_idx = jnp.sum(mask[i, :, 0])
        ref = np.array(y[i][: end_idx]).squeeze()
        deg = np.array(pred_y[i][: end_idx]).squeeze()
        loss += pesq(fs=16000, ref=ref, deg=deg, mode="wb")
    return loss / y.shape[0]


@eqx.filter_jit
def multi_res_stft_loss(
    y: Float[Array, "batch time feature"],
    pred_y: Float[Array, "batch time feature"],
    mask: Int[Array, "batch time feature"],
    fft_sizes: tuple[int] = (512, 1024, 2048),
    hop_sizes: tuple[int] = (50, 120, 240),
    win_lengths: tuple[int] = (240, 600, 1200),
) -> Float[Array, ""]:
    """Compute multi-resolution spectral loss."""
    y = (y * mask)[..., 0]
    pred_y = (pred_y * mask)[..., 0]

    total_loss = 0.0
    num_resolutions = len(fft_sizes)
    eps = jnp.finfo(y.dtype).eps

    for (
        fft_size,
        hop_size,
        win_length
    ) in zip(fft_sizes, hop_sizes, win_lengths, strict=True):
        _, _, y_stft = jax.scipy.signal.stft(
            y,
            nperseg=win_length,
            noverlap=win_length - hop_size,
            nfft=fft_size,
        )
        _, _, pred_y_stft = jax.scipy.signal.stft(
            pred_y,
            nperseg=win_length,
            noverlap=win_length - hop_size,
            nfft=fft_size,
        )

        y_mag = jnp.abs(y_stft)
        pred_y_mag = jnp.abs(pred_y_stft)

        sc_loss = jnp.mean(
                jnp.linalg.norm(y_mag - pred_y_mag, ord='fro', axis=(1, 2))
                / jnp.clip(jnp.linalg.norm(y_mag, ord='fro', axis=(1, 2)), a_min=eps)
        )
        lm_loss = jnp.mean(jnp.abs(jnp.log(pred_y_mag + eps) - jnp.log(y_mag + eps)))
        total_loss += (sc_loss + lm_loss)

    return total_loss / jnp.array(num_resolutions)
