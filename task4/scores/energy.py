import torch


def energy_unknownness(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits.float(), dim=1)
