from datetime import datetime

import equinox as eqx
import jax
import numpy as np
import optax
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import tasks.model as models
from tasks.dataloader import get_vb_demand_dataloaders
from tasks.train_util import (
    TrainConfig,
    TrainState,
    evaluate,
    load_checkpoint,
    prompt_device_precheck,
    save_checkpoint,
)


def train(train_cfg: TrainConfig):
    """Trains the model on the training dataset."""
    writer = SummaryWriter(train_cfg.log_dir)

    train_loader, test_loader = get_vb_demand_dataloaders(batch_size=train_cfg.batch_size)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    model = models.build_linoss_causal_spectral(subkey=subkey)
    state = eqx.nn.State(model=model)

    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.adam(train_cfg.learning_rate),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    ts = TrainState(
        model=model,
        opt_state=opt_state,
        model_state=state,
        key=key,
        tx=optimizer,
    )

    epoch_steps = len(train_loader)
    if train_cfg.resume_from_last_chkpt:
        ts = load_checkpoint(train_cfg.ckpt_dir, ts)
        global_step = int(ts.step)
        start_epoch = global_step // epoch_steps
        print(f"Resumed from step {global_step} (Starting at Epoch {start_epoch})")
    else:
        global_step = 0
        start_epoch = 0
        print("Starting training.")

    total_steps = train_cfg.num_epochs * epoch_steps
    for epoch in range(start_epoch, train_cfg.num_epochs):
        with tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch}/{train_cfg.num_epochs}",
            total=epoch_steps,
        ) as pbar:
            for local_step, item in pbar:
                if (epoch * epoch_steps + local_step) < global_step:
                    # skip to relevant batch
                    continue

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
                    evaluate(
                        ts,
                        test_loader=test_loader,
                        writer=writer,
                        num_samples=train_cfg.num_audio_samples,
                    )

                if (
                    global_step % train_cfg.save_interval == 0 and global_step > 0
                ) or is_last_step:
                    save_checkpoint(ts, train_cfg.ckpt_dir)

                global_step += 1
                local_step += 1

    print("Training complete.")
    return ts.model, ts.model_state


if __name__ == "__main__":
    prompt_device_precheck()

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_cfg = TrainConfig(
        batch_size=32,
        num_epochs=200,
        learning_rate=1e-5,
        log_interval=20,
        num_audio_samples=15,
        eval_interval=20,
        save_interval=1000,
        ckpt_dir=f"ckpts/latest",
        log_dir="runs/latest",
        resume_from_last_chkpt=False,
    )
    train(train_cfg)
