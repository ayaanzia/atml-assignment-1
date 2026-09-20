import torch


def msp_unknownness(logits: torch.Tensor) -> torch.Tensor:
    return 1.0 - logits.float().softmax(dim=1).amax(dim=1)
