"""Run a checkpointed SE-LinOSS model on one test sample and save a
spectrogram comparison figure (noisy / predicted / clean).

Usage:
    uv run python src/tasks/inference.py
    uv run python src/tasks/inference.py --ckpt ckpts/latest/165001.eqx --sample-idx 3
"""

import argparse
import logging
import os
import pathlib

logging.getLogger("jax").setLevel(logging.WARNING)

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.scipy import signal as jsignal
from scipy.io import wavfile

import tasks.model as models
from tasks.dataloader import get_vb_demand_dataloaders
from tasks.train_util import TrainState, infer


def _latest_ckpt(ckpt_dir: str) -> str:
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".eqx")]
    if not files:
        raise FileNotFoundError(f"No .eqx checkpoints in {ckpt_dir}")
    return os.path.join(ckpt_dir, max(files, key=lambda f: int(f.split(".")[0])))


def _stft_db(waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a normalized log-magnitude STFT tuned for visual inspection.

    Window 512 (32 ms @ 16 kHz) is the standard speech-analysis tradeoff:
    ~31 Hz true frequency resolution — fine enough to resolve harmonics of
    male and female voices — while still short enough to follow phoneme-scale
    transitions. Hop=64 (4 ms, 87.5% overlap) gives a finely sampled time
    axis; nfft=2048 zero-pads the display to ~7.8 Hz/bin for visual smoothness
    without altering the underlying resolution.
    """
    freqs, times, zxx = jsignal.stft(
        jnp.asarray(waveform), fs=sample_rate, nperseg=512, noverlap=448, nfft=2048
    )
    mag = jnp.abs(zxx)
    s_db = 20 * jnp.log10(jnp.clip(mag, min=1e-10))
    s_db = s_db - jnp.max(s_db)
    s_db = jnp.clip(s_db, min=-80)
    return np.asarray(freqs), np.asarray(times), np.asarray(s_db)


def _plot_comparison(
    noisy: np.ndarray,
    pred: np.ndarray,
    clean: np.ndarray,
    sample_rate: int,
    out_path: str,
    title_suffix: str = "",
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, sharey=True)
    titles = ("Noisy input", "Model prediction", "Clean target")
    last_img = None
    for ax, sig, title in zip(axes, (noisy, pred, clean), titles, strict=True):
        freqs, times, s_db = _stft_db(sig, sample_rate)
        last_img = ax.imshow(
            s_db,
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation=None,
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            vmin=-80,
            vmax=0,
        )
        ax.set_title(f"{title}{title_suffix}")
        ax.set_ylabel("Frequency (Hz)")
    axes[-1].set_xlabel("Time (s)")
    fig.colorbar(last_img, ax=axes, format="%+2.0f dB").set_label("Intensity (dB)")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_inference_model(
    ckpt_path: str, key: jax.Array
) -> tuple[eqx.Module, eqx.nn.State]:
    """Build a TrainState skeleton matching ``train.py`` and load weights.

    The saved checkpoint contains the full TrainState, including an opt_state
    produced by the chained optimizer used during training, so the skeleton
    must use the same chain or the leaf shapes won't match.
    """
    init_key, key = jax.random.split(key)
    model = models.build_se_linoss(subkey=init_key)
    state = eqx.nn.State(model=model)

    scheduler = optax.exponential_decay(
        init_value=5e-4, transition_steps=1, decay_rate=0.99, staircase=True
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.scale_by_adam(b1=0.8, b2=0.99),
        optax.add_decayed_weights(0.01),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    skeleton = TrainState(
        model=model,
        opt_state=opt_state,
        model_state=state,
        key=key,
        tx=optimizer,
        step=jnp.array(0, dtype=jnp.int32),
    )
    loaded: TrainState = eqx.tree_deserialise_leaves(ckpt_path, skeleton)
    return eqx.tree_inference(loaded.model, value=True), loaded.model_state


def _save_wavs(
    noisy: np.ndarray,
    pred: np.ndarray,
    clean: np.ndarray,
    sample_rate: int,
    fig_path: str,
) -> list[str]:
    """Write 16-bit PCM WAVs alongside the spectrogram figure.

    Predicted audio may exceed [-1, 1] slightly; clip before quantizing
    so we don't get integer wrap-around in the int16 cast.
    """
    stem, _ = os.path.splitext(fig_path)
    paths = []
    for name, wav in (("noisy", noisy), ("pred", pred), ("clean", clean)):
        out = f"{stem}_{name}.wav"
        pcm = np.clip(wav, -1.0, 1.0)
        pcm = (pcm * np.iinfo(np.int16).max).astype(np.int16)
        wavfile.write(out, sample_rate, pcm)
        paths.append(out)
    return paths


def _get_sample(test_loader, sample_idx: int) -> dict:
    for i, batch in enumerate(test_loader):
        if i == sample_idx:
            return batch
    raise IndexError(f"sample_idx {sample_idx} exceeds test set length")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to a specific .eqx checkpoint; defaults to newest in --ckpt-dir.",
    )
    parser.add_argument("--ckpt-dir", default="ckpts/latest")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--out_dir", default="exp")
    parser.add_argument("--out_file", default="spectrogram_comparison.png")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    ckpt_path = args.ckpt or _latest_ckpt(args.ckpt_dir)
    print(f"Loading checkpoint: {ckpt_path}")

    key = jax.random.PRNGKey(0)
    model, state = _load_inference_model(ckpt_path, key)

    _, test_loader = get_vb_demand_dataloaders(batch_size=1, num_workers=0)
    item = _get_sample(test_loader, args.sample_idx)

    noisy = item["arrays"]["noisy"].numpy()
    clean = item["arrays"]["clean"].numpy()
    mask = item["arrays"]["mask"].numpy()
    sample_id = item["meta"]["id"][0]

    output, _ = infer(model, noisy, state, key)
    pred = np.asarray(output.prediction)

    valid_len = int(mask[0, :, 0].sum())
    noisy_wav = noisy[0, :valid_len, 0]
    clean_wav = clean[0, :valid_len, 0]
    pred_wav = pred[0, :valid_len, 0]

    out_path = pathlib.Path(args.out_dir) / args.out_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _plot_comparison(
        noisy_wav,
        pred_wav,
        clean_wav,
        sample_rate=args.sample_rate,
        out_path=out_path.as_posix(),
        title_suffix=f"  —  {sample_id}",
    )
    print(f"Saved figure to {out_path.as_posix()}")

    wav_paths = _save_wavs(
        noisy_wav, pred_wav, clean_wav, sample_rate=args.sample_rate, fig_path=out_path.as_posix()
    )
    for p in wav_paths:
        print(f"Saved audio to {p}")


if __name__ == "__main__":
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")
    jax.config.update("jax_compilation_cache_dir", "/data5/baur/linaxSE/.jax_cache")
    main()
