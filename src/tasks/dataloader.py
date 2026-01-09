import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

__all__ = ["get_vb_demand_dataloaders"]


def _pad_to_length(tensor_list, target_length):
    """
    Pads a list of 1D tensors to target_length and stacks them.
    """
    padded_tensors = []
    for x in tensor_list:
        pad_size = target_length - x.shape[-1]
        if pad_size > 0:
            x = F.pad(x, (0, pad_size), mode='constant', value=0)
        else:
            x = x[:target_length]
        padded_tensors.append(x)

    return torch.stack(padded_tensors)


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

    clean_padded = _pad_to_length(clean_tensors, 32000)
    noisy_padded = _pad_to_length(noisy_tensors, 32000)

    mask = torch.zeros_like(clean_padded).bool()
    for i, src_audio in enumerate(clean_tensors):
        valid_len = min(src_audio.size(0), 32000)
        mask[i, :valid_len] = 1

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
