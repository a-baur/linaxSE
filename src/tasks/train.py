from datetime import datetime

import equinox as eqx
import jax
import numpy as np
import optax
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import tasks.model as models
from tasks.dataloader import get_vb_demand_dataloaders
from tasks.train_util import TrainState, TrainConfig, evaluate, save_checkpoint, prompt_device_precheck, load_checkpoint


def train(train_cfg: TrainConfig):
    """Trains the model on the training dataset."""
    writer = SummaryWriter()

    train_loader, test_loader = get_vb_demand_dataloaders(batch_size=train_cfg.batch_size)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    model = models.build_linoss(subkey=subkey)
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

    if train_cfg.resume_from_last_chkpt:
        ts = load_checkpoint(train_cfg.ckpt_dir, ts)
        start_epoch = ts.step // len(train_loader)
        print(f"Resumed from step {ts.step} (Starting at Epoch {start_epoch})")
    else:
        start_epoch = 0
        print("Starting training.")

    total_steps = train_cfg.num_epochs * len(train_loader)
    for epoch in range(start_epoch, train_cfg.num_epochs):
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

                is_last_step = ts.step == total_steps - 1
                if ts.step % train_cfg.log_interval == 0 or is_last_step:
                    pbar.set_postfix({"loss": f"{loss_value:.4f}"})
                    writer.add_scalar("Train/Loss", np.mean(loss_value).item(), ts.step)

                if ts.step % train_cfg.eval_interval == 0 or is_last_step:
                    evaluate(
                        ts,
                        test_loader=test_loader,
                        writer=writer,
                        num_samples=train_cfg.num_audio_samples
                    )

                if (ts.step % train_cfg.save_interval == 0 and ts.step > 0) or is_last_step:
                    save_checkpoint(ts, train_cfg.ckpt_dir)

    print("Training complete.")
    return ts.model, ts.model_state


if __name__ == "__main__":
    prompt_device_precheck()

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_cfg = TrainConfig(
        batch_size=32,
        num_epochs=200,
        learning_rate=1e-5,
        log_interval=1,
        num_audio_samples=15,
        eval_interval=500,
        save_interval=1000,
        ckpt_dir=f"ckpts/{date_str}",
        resume_from_last_chkpt=False,
    )
    train(train_cfg)
