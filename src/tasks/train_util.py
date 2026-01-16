import os
from dataclasses import dataclass

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
    batch_keys = jax.random.split(key, x.shape[0])
    pred_y, model_state = jax.vmap(
        model,
        axis_name="batch",
        in_axes=(0, None, 0),
        out_axes=(0, None),
    )(x, state, batch_keys)
    mse = jnp.sum(((pred_y - y) ** 2) * mask) / jnp.sum(mask)
    return mse, model_state


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
) -> float:
    mse = jnp.sum(((pred_y - y) ** 2) * mask) / jnp.sum(mask)
    return mse


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


@dataclass
class EvalMetric:
    mse: float
    pesq: float


class TrainState(eqx.Module):
    model: SSM
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
        for item in tqdm(test_loader, desc="Evaluating", leave=False):
            x = item["noisy"].numpy()
            y = item["clean"].numpy()
            mask = item["mask"].numpy()
            pred_y, model_state = infer(inference_model, x, self.model_state, self.key)
            cum_mse += mse_loss(y, pred_y, mask).item()
            cum_pesq += pesq_loss(y, pred_y, mask)
        return EvalMetric(
            mse=cum_mse / len(test_loader),
            pesq=cum_pesq / len(test_loader),
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
    num_blocks: int = 4
    hidden_size: int = 64
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