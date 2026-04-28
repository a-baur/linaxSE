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
    print_model_summary,
    prompt_device_precheck,
    save_checkpoint,
)


def train(train_cfg: TrainConfig):
    """Trains the model on the training dataset."""
    writer = SummaryWriter(train_cfg.log_dir)

    train_loader, test_loader = get_vb_demand_dataloaders(batch_size=train_cfg.batch_size)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    model = models.build_linoss_spectral(subkey=subkey)
    state = eqx.nn.State(model=model)

    scheduler = optax.exponential_decay(
        init_value=train_cfg.learning_rate,
        transition_steps=train_cfg.lr_transition_steps,
        decay_rate=train_cfg.lr_decay,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.scale_by_adam(
            b1=train_cfg.adam_beta1,
            b2=train_cfg.adam_beta2,
        ),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0),
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

    print_model_summary(ts.model)

    total_steps = train_cfg.num_epochs * epoch_steps
    for epoch in range(start_epoch, train_cfg.num_epochs):
        with tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch}/{train_cfg.num_epochs}",
            total=epoch_steps,
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

    print("Training complete.")
    return ts.model, ts.model_state


if __name__ == "__main__":
    prompt_device_precheck()

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_cfg = TrainConfig(
        batch_size=32,
        num_epochs=1000,
        learning_rate=1e-4,
        lr_transition_steps=10000,
        adam_beta1=0.8,
        adam_beta2=0.999,
        lr_decay=0.99,
        log_interval=20,
        num_audio_samples=15,
        eval_interval=1000,
        save_interval=5000,
        ckpt_dir="ckpts/latest",
        log_dir="runs/latest",
        resume_from_last_chkpt=False,
    )
    train(train_cfg)
