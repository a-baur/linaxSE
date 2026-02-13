import os
from dataclasses import dataclass

import torch
from torch.utils.tensorboard import SummaryWriter

import equinox as eqx
import jax
import numpy as np
import optax
from pesq import pesq
from jax import numpy as jnp
from jaxtyping import Float, Array, Int, PRNGKeyArray
from torch.utils.data import DataLoader
from tqdm import tqdm

from linax.models import SSM
from tasks import util


@eqx.filter_jit
def infer(
    model: SSM,
    x: Float[Array, "batch time feature"],
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> tuple[Float[Array, ""], eqx.nn.State]:
    batch_keys = jax.random.split(key, x.shape[0])
    pred_y, model_state = jax.vmap(
        model,
        axis_name="batch",
        in_axes=(0, None, 0),
        out_axes=(0, None),
    )(x, state, batch_keys)
    return pred_y, model_state


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
) -> Float[Array, ""]:
    """
    Computes Negative SI-SDR loss for a batch.

    Args:
        pred_y: (Batch, Time, 1) Estimated audio
        y: (Batch, Time, 1) Clean target audio
        mask: (Batch, Time, 1) Binary mask for valid lengths
        zero_mean: Whether to zero-mean the signals before computation

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

    return -jnp.mean(si_sdr)


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
def train_loss(
    model: SSM,
    x: Float[Array, "batch time feature"],
    y: Float[Array, "batch time feature"],
    mask: Int[Array, "batch time feature"],
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> tuple[Float[Array, ""], eqx.nn.State]:
    """ Infer and compute MSE loss in single function call for training efficiency. """
    pred_y, model_state = infer(model, x, state, key)
    loss = si_sdr_loss(y, pred_y, mask, zero_mean=True)
    return loss, model_state


@dataclass
class EvalMetric:
    mse: float
    pesq: float
    si_sdr: float


class TrainState(eqx.Module):
    model: SSM | eqx.Module
    opt_state: optax.OptState
    model_state: eqx.nn.State
    key: PRNGKeyArray
    tx: optax.GradientTransformation = eqx.field(static=True)

    @eqx.filter_jit
    def update(self, x, y, mask):
        key, train_key = jax.random.split(self.key)  # update key for next step
        (loss_val, new_model_state), grads = eqx.filter_value_and_grad(train_loss, has_aux=True)(
            self.model, x, y, mask, self.model_state, train_key
        )
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, eqx.filter(self.model, eqx.is_inexact_array)
        )
        new_model = eqx.apply_updates(self.model, updates)

        return TrainState(
            model=new_model,
            opt_state=new_opt_state,
            model_state=new_model_state,
            key=key,
            tx=self.tx,
        ), loss_val

    def evaluate(self, test_loader: DataLoader) -> EvalMetric:
        """Evaluates the model on the test dataset."""
        inference_model = eqx.tree_inference(self.model, value=True)
        cum_mse = 0
        cum_pesq = 0
        cum_sisdr = 0
        for item in tqdm(test_loader, desc="Evaluating", leave=False):
            x = item["noisy"].numpy()
            y = item["clean"].numpy()
            mask = item["mask"].numpy()
            pred_y, model_state = infer(inference_model, x, self.model_state, self.key)
            cum_mse += mse_loss(y, pred_y, mask).item()
            cum_pesq += pesq_loss(y, pred_y, mask)
            cum_sisdr += si_sdr_loss(y, pred_y, mask, zero_mean=True).item()
        return EvalMetric(
            mse=cum_mse / len(test_loader),
            pesq=cum_pesq / len(test_loader),
            si_sdr=cum_sisdr / len(test_loader),
        )

    @eqx.filter_jit
    def create_samples(
            self,
            test_loader: DataLoader,
            num_samples: int = 5,
    ) -> tuple[
        Float[Array, "batch time feature"],
        Float[Array, "batch time feature"],
        Float[Array, "batch time feature"],
        eqx.nn.State
    ]:
        """Generates enhanced samples from the model given noisy input."""
        inference_model = eqx.tree_inference(self.model, value=True)
        batch = next(iter(test_loader))
        x = batch["noisy"].numpy()[:num_samples]
        y = batch["clean"].numpy()[:num_samples]
        pred_y, model_state = jax.vmap(
            inference_model,
            axis_name="batch",
            in_axes=(0, None, 0),
            out_axes=(0, None),
        )(x, self.model_state, jax.random.split(self.key, x.shape[0]))
        return x, y, pred_y, model_state


@dataclass
class TrainConfig:
    batch_size: int = 16
    num_epochs: int = 1
    learning_rate: float = 1e-3
    log_interval: int = 50
    eval_interval: int = 200
    save_interval: int = 1000
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"

    def __post_init__(self):
        os.makedirs(self.ckpt_dir, exist_ok=True)


def evaluate(ts: TrainState, step: int, test_loader, writer: SummaryWriter):
    """Evaluates the model on the test dataset."""
    eval_metrics = ts.evaluate(test_loader)

    writer.add_scalar("Eval/MSE", eval_metrics.mse, step)
    writer.add_scalar("Eval/PESQ", eval_metrics.pesq, step)
    writer.add_scalar("Eval/SI_SDR", eval_metrics.si_sdr, step)

    num_samples = 5
    x, y, y_pred, _ = ts.create_samples(test_loader, num_samples=num_samples)
    if step == 0:
        for i in range(num_samples):
            writer.add_audio(
                f"Source/Sample_{i}",
                torch.from_numpy(np.array(x[i])).squeeze(),
                step,
                sample_rate=16000,
            )
            util.log_spectrogram(
                writer,
                f"Source/Spectrogram_Sample_{i}",
                np.array(x[i]).squeeze(),
                step,
                sample_rate=16000,
            )
            writer.add_audio(
                f"Target/Sample_{i}",
                torch.from_numpy(np.array(y[i])).squeeze(),
                step,
                sample_rate=16000,
            )
            util.log_spectrogram(
                writer,
                f"Target/Spectrogram_Sample_{i}",
                np.array(y[i]).squeeze(),
                step,
                sample_rate=16000,
            )

    for i, sample in enumerate(y_pred):
        writer.add_audio(
            f"Eval/Sample_{i}",
            torch.from_numpy(np.array(sample)).squeeze(),
            step,
            sample_rate=16000,
        )
        util.log_spectrogram(
            writer,
            f"Eval/Spectrogram_Sample_{i}",
            sample.squeeze(),
            step,
            sample_rate=16000,
        )


def save_checkpoint(model: eqx.Module, step: int, ckpt_dir: str):
    """Saves the model checkpoint."""
    ckpt_path = f"{ckpt_dir}/ckpt_step_{step}.eqx"
    eqx.tree_serialise_leaves(ckpt_path, model)
    print(f"\nSaved checkpoint at step {step} to {ckpt_path}")


def prompt_device_precheck():
    n_gpus = torch.cuda.device_count()

    if n_gpus > 0:
        devices = util.get_cuda_devices()
        devices = "\n".join(devices)
        proceed = input(
            f"proceed training on the following cuda devices (y/n)?\n{devices}\n"
        )
        if proceed.lower() == "n":
            raise KeyboardInterrupt
