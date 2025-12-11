import time

import equinox as eqx
import jax
import jax.numpy as jnp
import torch
from jaxtyping import Array, Float, Int, PRNGKeyArray

from linax.encoder import LinearEncoderConfig
from linax.models.linoss import LinOSSConfig
from linax.models.ssm import SSM
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

from linax.heads import RegressionHeadConfig


# Training configuration
BATCH_SIZE = 1  # Number of samples per batch
LEARNING_RATE = 3e-4  # AdamW learning rate
STEPS = 7500  # Total training steps.
PRINT_EVERY = 1500  # Evaluation frequency.
SEED = 5678  # Random seed for reproducibility
NUM_BLOCKS = 10  # Number of LinOSS blocks


def collate_fn_pad(batch):
    """
    Collate function to zero-pad audio arrays in a batch.

    Args:
        batch (list): List of dictionaries from the Dataset __getitem__.
                      Each item has 'id', 'clean', and 'noisy' keys.
    """
    clean_tensors = [torch.from_numpy(item['clean']['array']).float() for item in batch]
    noisy_tensors = [torch.from_numpy(item['noisy']['array']).float() for item in batch]

    ids = [item['id'] for item in batch]
    sampling_rates = [item['clean']['sampling_rate'] for item in batch]

    clean_padded = pad_sequence(clean_tensors, batch_first=True, padding_value=0.0)
    noisy_padded = pad_sequence(noisy_tensors, batch_first=True, padding_value=0.0)

    mask = torch.zeros_like(clean_padded).bool()
    for i, src_audio in enumerate(clean_tensors):
        mask[i, : src_audio.size(0)] = 1

    return {
        'id': ids,
        'clean': clean_padded[..., None],  # Add feature dimension
        'noisy': noisy_padded[..., None],
        'mask': mask[..., None],
        'sr': sampling_rates
    }

print("Loading dataset...")
ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
train_dataset, test_dataset = ds["train"], ds["test"]

key = jax.random.PRNGKey(SEED)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn_pad,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn_pad,
)

print("Building model...")
linoss_cfg = LinOSSConfig(
    num_blocks=NUM_BLOCKS,
    encoder_config=LinearEncoderConfig(in_features=1, out_features=64),
    head_config=RegressionHeadConfig(out_features=1, reduce=False),
)
key, subkey = jax.random.split(key, 2)
model = linoss_cfg.build(key=subkey)
state = eqx.nn.State(model=model)

print("Preparing a batch...")
batch = next(iter(train_loader))
# Add feature dimension
clean = batch['clean'].numpy()
noisy = batch['noisy'].numpy()
mask = batch['mask'].numpy()
print(" - Clean data shape:", clean.shape)
print(" - Noisy data shape:", noisy.shape)
print(" - Mask shape:", mask.shape)


def loss(
    model: SSM,
    x: Float[Array, "batch time"],
    y: Float[Array, "batch time"],
    mask: Int[Array, "batch time"],
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> Float[Array, ""]:
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
        Float[Array, ()]: Computed MSE loss.
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


print("Computing initial loss...")
start_time = time.time()
loss_value, _ = loss(
    model,
    clean,
    noisy,
    mask,
    state,
    key
)
end_time = time.time()
print(f"Loss computation time: {end_time - start_time:.4f} seconds")
print(f"Initial loss value: {loss_value:.4f}")
