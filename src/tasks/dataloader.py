import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

__all__ = ["get_vb_demand_dataloaders"]


def _collate_fn_vb_demand(batch) -> dict:
    """Collate function to zero-pad audio arrays in a batch.

    Args:
        batch (list): List of dictionaries from the Dataset __getitem__.
                      Each item has 'id', 'clean', and 'noisy' keys.
    """
    clean_tensors = [torch.from_numpy(item["clean"]["array"]).float() for item in batch]
    noisy_tensors = [torch.from_numpy(item["noisy"]["array"]).float() for item in batch]

    ids = [item["id"] for item in batch]
    sampling_rates = [item["clean"]["sampling_rate"] for item in batch]

    clean_padded = pad_sequence(clean_tensors, batch_first=True, padding_value=0.0)
    noisy_padded = pad_sequence(noisy_tensors, batch_first=True, padding_value=0.0)

    mask = torch.zeros_like(clean_padded).bool()
    for i, src_audio in enumerate(clean_tensors):
        mask[i, : src_audio.size(0)] = 1

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
