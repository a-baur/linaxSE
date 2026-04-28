import os
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from linax.models import SSM
from tasks import util
from tasks.loss import (
    l1_loss,
    mse_loss,
    multi_res_stft_loss,
    pesq_loss,
    si_sdr_loss,
    spectral_mag_loss,
)


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
def spectral_train_loss(
    model: SSM,
    x: Float[Array, "batch time feature"],
    y: Float[Array, "batch time feature"],
    mask: Int[Array, "batch time feature"],
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> tuple[Float[Array, ""], eqx.nn.State]:
    """Infer and compute MSE loss in single function call for training efficiency."""
    pred_c_mag, model_state = infer(model, x, state, key)
    loss = spectral_mag_loss(y, pred_c_mag, mask)
    return loss, model_state


@eqx.filter_jit
def train_loss(
    model: SSM,
    x: Float[Array, "batch time feature"],
    y: Float[Array, "batch time feature"],
    mask: Int[Array, "batch time feature"],
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> tuple[Float[Array, ""], eqx.nn.State]:
    """Infer and compute MSE loss in single function call for training efficiency."""
    pred_y, model_state = infer(model, x, state, key)
    loss = multi_res_stft_loss(y, pred_y, mask)
    # cplx = complex_stft_loss(y, pred_y, mask)
    # l1 = l1_loss(y, pred_y, mask)
    # loss = mrsl + cplx + l1
    return loss, model_state


@dataclass
class EvalMetric:
    label: str
    value: float

    def __init__(self, label: str, values: Float[Array, "batch"]) -> None:
        self.label = label
        self.value = values.mean().item()


class TrainState(eqx.Module):
    model: SSM | eqx.Module
    opt_state: optax.OptState
    model_state: eqx.nn.State
    key: PRNGKeyArray
    tx: optax.GradientTransformation = eqx.field(static=True)
    step: Array = eqx.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))

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
            step=self.step + 1,
        ), loss_val

    def evaluate(self, test_loader: DataLoader) -> list[EvalMetric]:
        """Evaluates the model on the test dataset."""
        inference_model = eqx.tree_inference(self.model, value=True)
        loss_funcs = {
            "L1": l1_loss,
            "MSE": mse_loss,
            "SI-SDR": si_sdr_loss,
            "PESQ": pesq_loss,
            "MultiResSTFT": multi_res_stft_loss,
        }
        losses = {name: [] for name in loss_funcs.keys()}
        for item in tqdm(test_loader, desc="Evaluating", leave=False):
            x = item["noisy"].numpy()
            y = item["clean"].numpy()
            mask = item["mask"].numpy()
            pred_y, model_state = infer(inference_model, x, self.model_state, self.key)
            for name, func in loss_funcs.items():
                loss_val = func(y, pred_y, mask)
                losses[name].append(loss_val)
        return [EvalMetric(name, jnp.array(vals)) for name, vals in losses.items()]

    def create_samples(
        self,
        test_loader: DataLoader,
        num_samples: int = 5,
    ) -> tuple[
        Float[Array, "batch time feature"],
        Float[Array, "batch time feature"],
        Float[Array, "batch time feature"],
        eqx.nn.State,
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
    lr_transition_steps: int = 1
    adam_beta1: float = 0.8
    adam_beta2: float = 0.999
    lr_decay: float = 0.99
    log_interval: int = 50
    eval_interval: int = 200
    save_interval: int = 1000
    num_audio_samples: int = 5
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"
    resume_from_last_chkpt: bool = False

    def __post_init__(self):
        os.makedirs(self.ckpt_dir, exist_ok=True)


def evaluate(ts: TrainState, test_loader, writer: SummaryWriter, num_samples: int = 5):
    """Evaluates the model on the test dataset.

    Args:
        ts: TrainState containing the model and state.
        test_loader: DataLoader for the test dataset.
        writer: SummaryWriter for logging.
        num_samples: Number of audio samples to log.
    """
    eval_metrics = ts.evaluate(test_loader)

    for metric in eval_metrics:
        writer.add_scalar(f"Eval/{metric.label}", metric.value, ts.step)

    x, y, y_pred, _ = ts.create_samples(test_loader, num_samples=num_samples)
    if ts.step == 0:
        for i in range(num_samples):
            writer.add_audio(
                f"Source/Sample_{i}",
                torch.from_numpy(np.array(x[i])).squeeze(),
                ts.step,
                sample_rate=16000,
            )
            util.log_spectrogram(
                writer,
                f"Source/Spectrogram_Sample_{i}",
                np.array(x[i]).squeeze(),
                ts.step,
                sample_rate=16000,
            )
            writer.add_audio(
                f"Target/Sample_{i}",
                torch.from_numpy(np.array(y[i])).squeeze(),
                ts.step,
                sample_rate=16000,
            )
            util.log_spectrogram(
                writer,
                f"Target/Spectrogram_Sample_{i}",
                np.array(y[i]).squeeze(),
                ts.step,
                sample_rate=16000,
            )

    for i, sample in enumerate(y_pred):
        writer.add_audio(
            f"Eval/Sample_{i}",
            torch.from_numpy(np.array(sample)).squeeze(),
            ts.step,
            sample_rate=16000,
        )
        util.log_spectrogram(
            writer,
            f"Eval/Spectrogram_Sample_{i}",
            sample.squeeze(),
            ts.step,
            sample_rate=16000,
        )


def save_checkpoint(ts: TrainState, ckpt_dir: str):
    """Saves the model checkpoint."""
    ckpt_path = f"{ckpt_dir}/{ts.step}.eqx"
    eqx.tree_serialise_leaves(ckpt_path, ts)
    # print(f"\nSaved checkpoint at step {ts.step} to {ckpt_path}")


def load_checkpoint(ckpt_dir: str, ts_skeleton: TrainState) -> TrainState:
    """Loads the complete training state from a checkpoint.

    Args:
        ckpt_dir: Path to the checkpoint directory containing the saved TrainState.
        ts_skeleton: A newly initialized TrainState to use as a structural template.
    """
    files = os.listdir(ckpt_dir)
    if not files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    latest_ckpt = max(files, key=lambda f: int(f.split(".")[0]))
    ckpt_path = os.path.join(ckpt_dir, latest_ckpt)

    print(f"Resuming from checkpoint: {latest_ckpt}")
    return eqx.tree_deserialise_leaves(ckpt_path, ts_skeleton)


def prompt_device_precheck():
    n_gpus = torch.cuda.device_count()

    if n_gpus > 0:
        devices = util.get_cuda_devices()
        devices = "\n".join(devices)
        proceed = input(f"proceed training on the following cuda devices (y/n)?\n{devices}\n")
        if proceed.lower() == "n":
            raise KeyboardInterrupt


def load_for_inference(
    model: eqx.Module, state: eqx.nn.State, key: PRNGKeyArray, ckpt_path: str
) -> tuple[eqx.Module, eqx.nn.State]:
    """Loads a checkpoint and prepares the model for inference."""
    # Dummy optimizer and opt_state to create the TrainState skeleton
    optimizer = optax.adam(1e-5)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    ts_skeleton = TrainState(
        model=model, opt_state=opt_state, model_state=state, key=key, tx=optimizer, step=0
    )

    ts_loaded: TrainState = eqx.tree_deserialise_leaves(ckpt_path, ts_skeleton)
    inference_model = eqx.tree_inference(ts_loaded.model, value=True)
    return inference_model, ts_loaded.model_state


def print_model_summary(model: eqx.Module):
    """Prints a concise overview of an Equinox model."""
    trainable, static = eqx.partition(model, eqx.is_inexact_array)

    all_leaves = jax.tree_util.tree_leaves(model)
    arrays = [x for x in all_leaves if eqx.is_array(x)]

    total_params = sum(x.size for x in arrays)
    trainable_params = sum(x.size for x in jax.tree_util.tree_leaves(trainable) if x is not None)
    total_size_mb = sum(x.nbytes for x in arrays) / (1024**2)

    print(f"\n{' Model Overview ':=^35}")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Params:     {trainable_params:,}")
    print(f"Non-trainable Params: {total_params - trainable_params:,}")
    print(f"Model Size:           {total_size_mb:.2f} MB")
    print(f"{'=' * 35}\n")
