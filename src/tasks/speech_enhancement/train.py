import equinox as eqx
import jax
import optax
from tqdm import tqdm

from tasks.dataloader import get_vb_demand_dataloaders
from tasks.model import build_linoss_model
from tasks.train_util import TrainState, TrainConfig


def train(train_cfg: TrainConfig):
    """Trains the model on the training dataset."""
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

    num_steps = 0
    for epoch in range(train_cfg.num_epochs):
        with tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch}/{train_cfg.num_epochs}",
            total=len(train_loader)
        ) as pbar:
            for step, item in pbar:
                x = item["noisy"].numpy()
                y = item["clean"].numpy()
                mask = item["mask"].numpy()
                ts, loss_value = ts.update(x, y, mask)

                is_last_step = step == num_steps - 1
                if step % train_cfg.log_interval == 0 or is_last_step:
                    pbar.set_postfix({"loss": f"{loss_value:.4f}"})

                if (step % train_cfg.eval_interval == 0 and step > 0) or is_last_step:
                    eval_loss = ts.evaluate(test_loader)
                    print(f"\nEvaluation loss at step {step}: {eval_loss:.4f}")

                if (step % train_cfg.save_interval == 0 and step > 0) or is_last_step:
                    ckpt_path = f"{train_cfg.ckpt_dir}/ckpt_step_{step}.eqx"
                    eqx.tree_serialise_leaves(ckpt_path, ts.model)
                    print(f"\nSaved checkpoint at step {step} to {ckpt_path}")

                num_steps += 1

    print("Training complete.")
    return ts.model, ts.model_state


if __name__ == "__main__":
    train_cfg = TrainConfig(
        num_blocks=2,
        hidden_size=16,
        batch_size=4,
        num_epochs=1,
        learning_rate=1e-3,
        log_interval=1,
        eval_interval=5,
        save_interval=5,
        ckpt_dir="checkpoints",
    )
    train(train_cfg)
