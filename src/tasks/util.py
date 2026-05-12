import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from jax.scipy import signal
from torch.utils.tensorboard import SummaryWriter


def get_cuda_devices() -> list[str]:
    """Get list of cuda devices available for training."""
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    n_gpus = torch.cuda.device_count()
    if cuda_visible_devices is not None:
        gpu_ids = list(map(int, cuda_visible_devices.split(",")))
    else:
        gpu_ids = list(range(n_gpus))

    device_info = []
    for idx, i in zip(gpu_ids, range(n_gpus)):
        name = torch.cuda.get_device_name(i)
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mem_free, mem_total = mem_free / 1024 ** 3, mem_total / 1024 ** 3
        mem_usage = mem_total - mem_free
        percent = mem_usage / mem_total
        info = f"{name} [gpu:{idx} | cuda:{i} | utilization: {percent:7.2%} ({mem_usage:4.1f}GB/{mem_total:4.1f}GB)]"
        device_info.append(info)

    return device_info


def create_spec(
        waveform: jnp.ndarray,
        sample_rate: int = 16000,
        title: str = "Spectrogram",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))

    # Ensure 1D array
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    # 1. Compute STFT natively in JAX
    freqs, times, Zxx = signal.stft(
        waveform,
        fs=sample_rate,
        nperseg=512,  # 32 ms window
        noverlap=384,  # 8 ms hop
        nfft=1024,
    )

    # 2. Magnitude and dB conversion
    mag = jnp.abs(Zxx)
    s_db = 20 * jnp.log10(jnp.clip(mag, min=1e-10))

    # Reference to max and apply -80 dB floor
    s_db = s_db - jnp.max(s_db)
    s_db = jnp.clip(s_db, min=-80)

    # Move to CPU for plotting to prevent Matplotlib warnings
    s_db_np = np.asarray(s_db)
    times_np = np.asarray(times)
    freqs_np = np.asarray(freqs)

    # 3. Plot using imshow
    img = ax.imshow(
        s_db_np,
        aspect='auto',
        origin='lower',
        cmap='magma',
        extent=[times_np[0], times_np[-1], freqs_np[0], freqs_np[-1]],
        vmin=-80,
        vmax=0
    )

    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB").set_label('Intensity (dB)')

    return fig


def log_spectrogram(
        writer: SummaryWriter,
        tag: str,
        waveform: np.ndarray,
        step: int,
        sample_rate=16000,
):
    """Computes and logs a spectrogram figure to TensorBoard."""
    fig = create_spec(waveform, sample_rate=sample_rate, title=tag)
    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)
