from typing import Callable, Any

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

__all__ = ["get_vb_demand_dataloaders"]


def _get_vb_demand_collate_fn(sample_length: int, random_slice: bool = False) -> Callable[[Any], dict]:

    def _collate_fn(batch) -> dict:
        """Collate function to zero-pad audio arrays in a batch.

        Args:
            batch (list): List of dictionaries from the Dataset __getitem__.
                          Each item has 'id', 'clean', and 'noisy' keys.
        """

        clean_tensors = [torch.from_numpy(item["clean"]["array"]).float() for item in batch]
        noisy_tensors = [torch.from_numpy(item["noisy"]["array"]).float() for item in batch]

        ids = [item["id"] for item in batch]
        sampling_rates = [item["clean"]["sampling_rate"] for item in batch]

        clean_padded = torch.zeros(len(batch), sample_length)
        noisy_padded = torch.zeros(len(batch), sample_length)
        mask = torch.zeros_like(clean_padded).bool()

        # Pad sequences to fixed length and create mask
        for i, (c, n) in enumerate(zip(clean_tensors, noisy_tensors)):
            total = c.shape[0]
            if random_slice and total > sample_length:
                start = int(torch.randint(0, total - sample_length + 1, (1,)).item())
            else:
                start = 0
            L = min(total - start, sample_length)
            mask[i, :L] = 1
            clean_padded[i, :L] = c[start : start + L]
            noisy_padded[i, :L] = n[start : start + L]

        return {
            "id": ids,
            "clean": clean_padded[..., None],  # Add feature dimension
            "noisy": noisy_padded[..., None],
            "mask": mask[..., None],
            "sr": sampling_rates,
        }

    return _collate_fn


def get_vb_demand_dataloaders(
    batch_size: int,
    sample_length: int = 32000,
    shuffle_test_data: bool = False,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
    train_dataset, test_dataset = ds["train"], ds["test"]
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
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
