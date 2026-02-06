import equinox as eqx
import jax
import numpy as np
import optax
import torch
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from tasks.dataloader import get_vb_demand_dataloaders
from tasks.model import build_linoss_time
from tasks.train_util import TrainState, TrainConfig, evaluate, save_checkpoint, prompt_device_precheck


def train(train_cfg: TrainConfig):
    """Trains the model on the training dataset."""
    writer = SummaryWriter()

    train_loader, test_loader = get_vb_demand_dataloaders(batch_size=train_cfg.batch_size)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    model = build_linoss_time(subkey=subkey)
    state = eqx.nn.State(model=model)

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
                    writer.add_scalar("Train/Loss", np.mean(loss_value).item(), global_step)

                if global_step % train_cfg.eval_interval == 0 or is_last_step:
                    evaluate(ts, global_step, test_loader, writer)

                if (global_step % train_cfg.save_interval == 0 and global_step > 0) or is_last_step:
                    save_checkpoint(ts.model, global_step, train_cfg.ckpt_dir)

                global_step += 1

    print("Training complete.")
    return ts.model, ts.model_state


if __name__ == "__main__":
    prompt_device_precheck()
    train_cfg = TrainConfig(
        batch_size=32,
        num_epochs=200,
        learning_rate=1e-5,
        log_interval=1,
        eval_interval=500,
        save_interval=1000,
        ckpt_dir="checkpoints",
    )
    train(train_cfg)
