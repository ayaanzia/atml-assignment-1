import torch


def mls_unknownness(logits: torch.Tensor) -> torch.Tensor:
    return -logits.float().amax(dim=1)
