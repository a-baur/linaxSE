from multiprocessing import Pool, cpu_count

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
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
def spectral_mag_loss(
    y: Float[Array, "batch time feature"],
    pred_c_mag: Float[Array, "batch time feature"],
    mask: Int[Array, "batch time feature"],
) -> Float[Array, ""]:
    spectral_mask = jax.image.resize(
        mask, shape=(mask.shape[0], pred_c_mag.shape[1], mask.shape[2]), method="nearest"
    )
    _, _, Zxx = jax.scipy.signal.stft(
        y.squeeze(),
        nperseg=512,
        noverlap=512 - 256,
        nfft=512,
    )

    Zxx = jnp.swapaxes(Zxx, -1, -2)
    y_mag = jnp.abs(Zxx)
    y_c_mag = jnp.log1p(y_mag)
    l1 = jnp.sum(jnp.abs(pred_c_mag - y_c_mag) * spectral_mask) / jnp.sum(spectral_mask)
    return l1


@eqx.filter_jit
def si_sdr_loss(
    y: Float[Array, "batch time feature"],
    pred_y: Float[Array, "batch time feature"],
    mask: Int[Array, "batch time feature"],
    zero_mean: bool = True,
    invert: bool = False,
) -> Float[Array, ""]:
    """Computes Negative SI-SDR loss for a batch.

    Args:
        pred_y: (Batch, Time, 1) Estimated audio
        y: (Batch, Time, 1) Clean target audio
        mask: (Batch, Time, 1) Binary mask for valid lengths
        zero_mean: Whether to zero-mean the signals before computation
        invert (bool): Whether to return negative SI-SDR for minimization (default False)

    Returns:
        Negative SI-SDR loss
    """
    pred_y = pred_y.squeeze(-1) * mask.squeeze(-1)
    y = y.squeeze(-1) * mask.squeeze(-1)

    if zero_mean:
        pred_y = pred_y - jnp.mean(pred_y, axis=-1, keepdims=True)
        y = y - jnp.mean(y, axis=-1, keepdims=True)

    eps = jax.numpy.finfo(pred_y.dtype).eps

    alpha = (jnp.sum(pred_y * y, axis=-1, keepdims=True) + eps) / (
        jnp.sum(y**2, axis=-1, keepdims=True) + eps
    )
    target_scaled = alpha * y

    noise = target_scaled - pred_y

    target_pow = jnp.sum(target_scaled**2, axis=1) + eps
    noise_pow = jnp.sum(noise**2, axis=1) + eps
    si_sdr = 10.0 * jnp.log10(target_pow / noise_pow)

    return jnp.mean(si_sdr) * (-1 if invert else 1)


def _pesq_pair(ref_deg: tuple[np.ndarray, np.ndarray]) -> float:
    ref, deg = ref_deg
    try:
        return pesq(fs=16000, ref=ref, deg=deg, mode="wb")
    except ValueError:
        return 0.0


def pesq_loss(
    y: Float[Array, "batch time feature"],
    pred_y: Float[Array, "batch time feature"],
    mask: Int[Array, "batch time feature"],
) -> Float[Array, ""]:
    batch_size = y.shape[0]
    end_indices = np.asarray(jnp.sum(mask[..., 0], axis=1))
    y_np = np.asarray(y)
    pred_y_np = np.asarray(pred_y)

    pairs = [
        (y_np[i, : end_indices[i]].squeeze(), pred_y_np[i, : end_indices[i]].squeeze())
        for i in range(batch_size)
    ]

    with Pool(min(batch_size, cpu_count())) as pool:
        scores = pool.map(_pesq_pair, pairs)
    return sum(scores) / batch_size


def _anti_wrap(x: Array) -> Array:
    """SEMamba's anti-wrapping operator: |x - 2π·round(x / 2π)|.

    Maps an angular difference into [0, π], so a phase error close to ±2π
    is treated as the small wrap-around it actually is, not a large error.
    """
    return jnp.abs(x - jnp.round(x / (2 * jnp.pi)) * 2 * jnp.pi)


def phase_losses(
    phase_pred: Float[Array, "batch frames bins"],
    phase_target: Float[Array, "batch frames bins"],
    mask: Float[Array, "batch frames 1"] | None = None,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Anti-wrapping phase losses (SEMamba / MP-SENet).

    Returns (IP, GD, IAF):
      - IP: per-bin phase error
      - GD: phase derivative along the frequency axis (group delay)
      - IAF: phase derivative along the time axis (instantaneous angular frequency)
    """
    ip_diff = _anti_wrap(phase_pred - phase_target)
    gd_diff = _anti_wrap(jnp.diff(phase_pred, axis=-1) - jnp.diff(phase_target, axis=-1))
    iaf_diff = _anti_wrap(jnp.diff(phase_pred, axis=-2) - jnp.diff(phase_target, axis=-2))

    if mask is None:
        return jnp.mean(ip_diff), jnp.mean(gd_diff), jnp.mean(iaf_diff)

    eps = jnp.finfo(phase_pred.dtype).eps
    ip = jnp.sum(ip_diff * mask) / (jnp.sum(mask) * ip_diff.shape[-1] + eps)
    gd = jnp.sum(gd_diff * mask) / (jnp.sum(mask) * gd_diff.shape[-1] + eps)
    iaf_mask = mask[..., :-1, :] * mask[..., 1:, :]
    iaf = jnp.sum(iaf_diff * iaf_mask) / (jnp.sum(iaf_mask) * iaf_diff.shape[-1] + eps)
    return ip, gd, iaf


@eqx.filter_jit
def spectral_losses(
    y: Float[Array, "batch time feature"],
    mag_pred: Float[Array, "batch frames bins"],
    phase_pred: Float[Array, "batch frames bins"],
    mask: Int[Array, "batch time feature"],
    n_fft: int = 400,
    hop_length: int = 100,
    win_length: int = 400,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Magnitude, anti-wrapping phase, and complex losses against a single
    STFT of the clean target — using the model's mag_pred / phase_pred
    directly instead of re-STFT'ing the time-domain prediction.

    Returns (mag_l, pha_l, com_l) with pha_l = IP + GD + IAF.
    """
    p = 0.3
    n_frames = mag_pred.shape[-2]
    n_bins = mag_pred.shape[-1]
    spec_mask = jax.image.resize(
        mask.astype(jnp.float32),
        shape=(mask.shape[0], n_frames, 1),
        method="nearest",
    )
    eps = jnp.finfo(mag_pred.dtype).eps
    norm = jnp.sum(spec_mask) * n_bins + eps

    y_masked = (y * mask)[..., 0]
    _, _, Y = jax.scipy.signal.stft(
        y_masked,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
    )
    Y = jnp.swapaxes(Y, -1, -2)  # [batch, frames, bins]
    y_mag = jnp.abs(Y) ** p
    y_phase = jnp.angle(Y)

    mag_l = jnp.sum(((mag_pred - y_mag) ** 2) * spec_mask) / norm

    ip, gd, iaf = phase_losses(phase_pred, y_phase, spec_mask)
    pha_l = ip + gd + iaf

    pred_real = mag_pred * jnp.cos(phase_pred)
    pred_imag = mag_pred * jnp.sin(phase_pred)
    Y_real = y_mag * jnp.cos(y_phase)
    Y_imag = y_mag * jnp.sin(y_phase)
    com_l = jnp.sum(((pred_real - Y_real) ** 2 + (pred_imag - Y_imag) ** 2) * spec_mask) / norm

    return mag_l, pha_l, com_l


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

    for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths, strict=True):
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
            jnp.linalg.norm(y_mag - pred_y_mag, ord="fro", axis=(1, 2))
            / jnp.clip(jnp.linalg.norm(y_mag, ord="fro", axis=(1, 2)), a_min=eps)
        )
        lm_loss = jnp.mean(jnp.abs(jnp.log(pred_y_mag + eps) - jnp.log(y_mag + eps)))
        total_loss += sc_loss + lm_loss

    return total_loss / jnp.array(num_resolutions)
