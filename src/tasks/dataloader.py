from typing import Callable, Any

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

__all__ = ["get_vb_demand_dataloaders"]


def _get_vb_demand_collate_fn(sample_length: int) -> Callable[[Any], dict]:

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
            L = min(c.shape[0], sample_length)
            mask[i, :L] = 1
            clean_padded[i, :L] = c[:L]
            noisy_padded[i, :L] = n[:L]

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
) -> tuple[DataLoader, DataLoader]:
    ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
    train_dataset, test_dataset = ds["train"], ds["test"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_get_vb_demand_collate_fn(sample_length),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test_data,
        collate_fn=_get_vb_demand_collate_fn(sample_length),
    )
    return train_loader, test_loader
