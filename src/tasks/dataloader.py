from typing import Callable, Any

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

__all__ = ["get_vb_demand_dataloaders"]


def _get_vb_demand_collate_fn(sample_length: int, random_slice: bool = False) -> Callable[[Any], dict]:
    def _collate_fn(batch) -> dict:
        B = len(batch)

        # Pre-allocate directly in the final required shape
        clean_padded = torch.zeros((B, sample_length, 1), dtype=torch.float32)
        noisy_padded = torch.zeros((B, sample_length, 1), dtype=torch.float32)
        mask = torch.zeros((B, sample_length, 1), dtype=torch.bool)

        ids = []
        sampling_rates = []

        for i, item in enumerate(batch):
            ids.append(item["id"])
            sampling_rates.append(item["clean"]["sampling_rate"])

            c = item["clean"]["array"]
            n = item["noisy"]["array"]
            total = c.shape[0]

            # Use numpy for random logic since arrays are now numpy
            if random_slice and total > sample_length:
                start = np.random.randint(0, total - sample_length + 1)
            else:
                start = 0

            L = min(total - start, sample_length)

            # Fill directly from numpy slices
            mask[i, :L, 0] = True
            clean_padded[i, :L, 0] = torch.from_numpy(c[start: start + L])
            noisy_padded[i, :L, 0] = torch.from_numpy(n[start: start + L])

        return {
            # Data for JAX device_put
            "arrays": {
                "clean": clean_padded,
                "noisy": noisy_padded,
                "mask": mask,
            },
            # Metadata to keep in Python
            "meta": {
                "id": ids,
                "sr": sampling_rates,
            }
        }

    return _collate_fn


def get_vb_demand_dataloaders(
        batch_size: int,
        sample_length: int = 32000,
        shuffle_test_data: bool = False,
        num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")

    # STEP 1: Format as numpy to eliminate Python object overhead
    ds = ds.with_format("numpy")

    train_dataset, test_dataset = ds["train"], ds["test"]

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,  # Set to False for JAX, you must use jax.device_put asynchronously
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        collate_fn=_get_vb_demand_collate_fn(sample_length, random_slice=True),
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=shuffle_test_data,
        drop_last=True,
        collate_fn=_get_vb_demand_collate_fn(sample_length, random_slice=False),
        **loader_kwargs,
    )

    return train_loader, test_loader
