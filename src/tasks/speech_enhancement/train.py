import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray
from torch.utils.data import DataLoader
from tqdm import tqdm

from linax.models.ssm import SSM

from src.tasks.dataloader import get_vb_demand_dataloaders
from src.tasks.model import build_linoss_model


@eqx.filter_jit
def loss(
    model: SSM,
    x: Float[Array, "batch time feature"],
    y: Float[Array, "batch time feature"],
    mask: Int[Array, "batch time feature"],
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> tuple[Float[Array, ()], eqx.nn.State]:
    """
    Computes the Mean Squared Error loss between the model's output and the target.

    Args:
        model (SSM): The LinOSS model.
        x (Float[Array, "batch time"]): Noisy input audio batch.
        y (Float[Array, "batch time"]): Clean target audio batch.
        mask (Int[Array, "batch time"]): Mask indicating valid audio samples.
        state (eqx.nn.State): State of the model.
        key (PRNGKeyArray): JAX random key.

    Returns:
        tuple[Float[Array, ()], eqx.nn.State]: Computed MSE loss and updated model state.
    """
    batch_keys = jax.random.split(key, x.shape[0])
    pred_y, model_state = jax.vmap(
        model,
        axis_name="batch",
        in_axes=(0, None, 0),
        out_axes=(0, None),
    )(x, state, batch_keys)
    mse = jnp.sum(((pred_y - y) ** 2) * mask) / jnp.sum(mask)
    return mse, model_state


def evaluate(
    model: SSM,
    test_loader: DataLoader,
    state: eqx.nn.State,
    key: PRNGKeyArray,
):
    """Evaluates the model on the test dataset."""
    inference_model = eqx.tree_inference(model, value=True)
    avg_loss = 0
    for item in tqdm(test_loader, desc="Evaluating"):
        x = item["clean"].numpy()
        y = item["noisy"].numpy()
        mask = item["mask"].numpy()
        avg_loss += loss(inference_model, x, y, mask, state, key)[0]
    return avg_loss / len(test_loader)


@eqx.filter_jit
def train_step(
    model: SSM,
    optimizer: optax.GradientTransformation,
    state: eqx.nn.State,
    opt_state: optax.OptState,
    x: Float[Array, "batch time feature"],
    y: Float[Array, "batch time feature"],
    mask: Int[Array, "batch time feature"],
    key: PRNGKeyArray,
) -> tuple[SSM, eqx.nn.State, optax.OptState, Float[Array, ()]]:
    (loss_value, new_state), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
        model, x, y, mask, state, key
    )
    updates, new_opt_state = optimizer.update(
        grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
    )
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, new_opt_state, loss_value


def train(
    model: SSM,
    optimizer: optax.GradientTransformation,
    train_loader: DataLoader,
    test_loader: DataLoader,
    state: eqx.nn.State,
    key: PRNGKeyArray,
    num_steps: int,
    log_interval: int = 1,
    eval_interval: int = 100,
):
    """Trains the model on the training dataset."""
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def infinite_train_loader():
        while True:
            yield from train_loader

    for step, item in zip(tqdm(range(num_steps), desc="Training"), infinite_train_loader()):
        x = item["clean"].numpy()
        y = item["noisy"].numpy()
        mask = item["mask"].numpy()
        key, subkey = jax.random.split(key)
        model, state, opt_state, loss_value = train_step(
            model, optimizer, state, opt_state, x, y, mask, subkey
        )

        is_last_step = step == num_steps - 1
        if step % log_interval == 0 or is_last_step:
            print(f"Step {step}, Loss: {loss_value:.4f}")

        if (step % eval_interval == 0 and step > 0) or is_last_step:
            eval_loss = evaluate(model, test_loader, state, key)
            print(f"Step {step}, Eval Loss: {eval_loss:.4f}")

    return model, state


def main():
    # Hyperparameters
    batch_size = 16
    num_steps = 1000
    learning_rate = 1e-3
    num_blocks = 4
    hidden_size = 64

    # Prepare data loaders
    train_loader, test_loader = get_vb_demand_dataloaders(batch_size=batch_size)

    # Initialize model and state
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    model, state = build_linoss_model(num_blocks=num_blocks, hidden_size=hidden_size, subkey=subkey)

    # Initialize optimizer
    optimizer = optax.adam(learning_rate)

    # Train the model
    model, state = train(
        model,
        optimizer,
        train_loader,
        test_loader,
        state,
        key,
        num_steps=num_steps,
        log_interval=50,
        eval_interval=200,
        )
    return model, state


if __name__ == "__main__":
    main()