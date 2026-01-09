import equinox as eqx
import jax
import optax
import torch
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from tasks import util
from tasks.dataloader import get_vb_demand_dataloaders
from tasks.model import build_linoss_model
from tasks.train_util import TrainState, TrainConfig


def evaluate(ts: TrainState, step: int, test_loader, writer: SummaryWriter):
    """Evaluates the model on the test dataset."""
    eval_loss = ts.evaluate(test_loader)
    writer.add_scalar("Eval/Loss", eval_loss, step)

    samples, _ = ts.create_samples(test_loader, num_samples=5)
    for i, sample in enumerate(samples):
        writer.add_audio(
            f"Eval/Sample_{i}",
            sample.squeeze(),
            0,
            sample_rate=16000,
        )


def save_checkpoint(model: eqx.Module, step: int, ckpt_dir: str):
    """Saves the model checkpoint."""
    ckpt_path = f"{ckpt_dir}/ckpt_step_{step}.eqx"
    eqx.tree_serialise_leaves(ckpt_path, model)
    print(f"\nSaved checkpoint at step {step} to {ckpt_path}")


def train(train_cfg: TrainConfig):
    """Trains the model on the training dataset."""
    writer = SummaryWriter()

    train_loader, test_loader = get_vb_demand_dataloaders(batch_size=train_cfg.batch_size)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    model, state = build_linoss_model(
        num_blocks=train_cfg.num_blocks,
        hidden_size=train_cfg.hidden_size,
        subkey=subkey,
    )

    optimizer = optax.adam(train_cfg.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    ts = TrainState(
        model=model,
        opt_state=opt_state,
        model_state=state,
        key=key,
        tx=optimizer,
    )

    total_steps = train_cfg.num_epochs * len(train_loader)
    global_step = 0
    for epoch in range(train_cfg.num_epochs):
        with tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch}/{train_cfg.num_epochs}",
            total=len(train_loader)
        ) as pbar:
            for _, item in pbar:
                item: dict

                x = item["noisy"].numpy()
                y = item["clean"].numpy()
                mask = item["mask"].numpy()
                ts, loss_value = ts.update(x, y, mask)

                is_last_step = global_step == total_steps - 1
                if global_step % train_cfg.log_interval == 0 or is_last_step:
                    pbar.set_postfix({"loss": f"{loss_value:.4f}"})
                    writer.add_scalar("Train/Loss", loss_value, global_step)

                if (global_step % train_cfg.eval_interval == 0 and global_step > 0) or is_last_step:
                    evaluate(ts, global_step, test_loader, writer)

                if (global_step % train_cfg.save_interval == 0 and global_step > 0) or is_last_step:
                    save_checkpoint(ts.model, global_step, train_cfg.ckpt_dir)

                global_step += 1

    print("Training complete.")
    return ts.model, ts.model_state


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    if n_gpus > 0:
        devices = util.get_cuda_devices()
        devices = "\n".join(devices)
        proceed = input(
            f"proceed training on the following cuda devices (y/n)?\n{devices}\n"
        )
        if proceed.lower() == "n":
            raise KeyboardInterrupt

    train_cfg = TrainConfig(
        num_blocks=2,
        hidden_size=16,
        batch_size=32,
        num_epochs=20,
        learning_rate=1e-3,
        log_interval=100,
        eval_interval=200,
        save_interval=400,
        ckpt_dir="checkpoints",
    )
    train(train_cfg)
