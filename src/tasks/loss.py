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
