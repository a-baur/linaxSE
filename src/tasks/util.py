import torch


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
        mem_free, mem_total = mem_free / 1024**3, mem_total / 1024**3
        mem_usage = mem_total - mem_free
        percent = mem_usage / mem_total
        info = f"{name} [gpu:{idx} | cuda:{i} | utilization: {percent:7.2%} ({mem_usage:4.1f}GB/{mem_total:4.1f}GB)]"
        device_info.append(info)

    return device_info