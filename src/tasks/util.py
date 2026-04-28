import os

import matplotlib.pyplot as plt
import numpy as np
import torch
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
        waveform: np.ndarray,
        sample_rate=16000,
        title="Spectrogram",
) -> plt.Figure:
    """Creates a spectrogram figure from a waveform."""
    spec = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=None)(waveform)
    mag = torch.abs(spec.squeeze())

    # 2. Convert to dB, reference max to 0 dB, and apply -80 dB floor
    s_db = 20 * torch.log10(torch.clamp(mag, min=1e-10))
    s_db = s_db - s_db.max()
    s_db = torch.clamp(s_db, min=-80)

    # 3. Plot using matplotlib.pyplot.imshow
    duration = waveform.shape[-1] / sample_rate
    img = ax.imshow(
        s_db.numpy(),
        aspect='auto',
        origin='lower',
        cmap='magma',
        extent=[0, duration, 0, sample_rate / 2],
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
