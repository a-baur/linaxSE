import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

__all__ = ["get_vb_demand_dataloaders"]


def _collate_fn_vb_demand(batch) -> dict:
    """Collate function to zero-pad audio arrays in a batch.

    Args:
        batch (list): List of dictionaries from the Dataset __getitem__.
                      Each item has 'id', 'clean', and 'noisy' keys.
    """
    MAX_LEN = 32000

    clean_tensors = [torch.from_numpy(item["clean"]["array"]).float() for item in batch]
    noisy_tensors = [torch.from_numpy(item["noisy"]["array"]).float() for item in batch]

    ids = [item["id"] for item in batch]
    sampling_rates = [item["clean"]["sampling_rate"] for item in batch]

    clean_padded = torch.zeros(len(batch), MAX_LEN)
    noisy_padded = torch.zeros(len(batch), MAX_LEN)
    mask = torch.zeros_like(clean_padded).bool()

    # Pad sequences to fixed length and create mask
    for i, (c, n) in enumerate(zip(clean_tensors, noisy_tensors)):
        L = min(c.shape[0], MAX_LEN)
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


def get_vb_demand_dataloaders(
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
    train_dataset, test_dataset = ds["train"], ds["test"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn_vb_demand,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn_vb_demand,
    )
    return train_loader, test_loader
