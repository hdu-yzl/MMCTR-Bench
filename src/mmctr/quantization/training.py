"""Small explicit training helpers for quantization premodels."""

from typing import Mapping

import torch

from mmctr.core import ContractError

from .psrq import PSRQPretrainer


def fit_psrq(
    model: PSRQPretrainer,
    features: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """Fit PSRQ over an item feature table and return final mean training loss."""

    if epochs <= 0 or batch_size <= 0:
        raise ContractError("PSRQ epochs and batch size must be positive")
    if set(features) != set(model.modalities):
        raise ContractError("PSRQ training features must match configured modalities")
    sizes = {name: values.shape[0] for name, values in features.items()}
    if len(set(sizes.values())) != 1 or not sizes:
        raise ContractError("PSRQ training feature tables must share a non-zero row count")
    sample_count = next(iter(sizes.values()))
    if sample_count == 0:
        raise ContractError("PSRQ training feature table is empty")
    if min(batch_size, sample_count) < model.codebook_size:
        raise ContractError("first PSRQ batch must contain at least codebook_size rows")

    tensors = {name: values.to(device) for name, values in features.items()}
    model.to(device)
    final_loss = 0.0
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            batch = {name: values[start:end] for name, values in tensors.items()}
            optimizer.zero_grad()
            output = model(batch)
            loss = output.total_loss()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            batches += 1
        final_loss = epoch_loss / batches
    return final_loss


__all__ = ["fit_psrq"]
