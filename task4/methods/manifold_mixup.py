"""Vectorized different-class pairing and layer-2 manifold mixup."""
from __future__ import annotations

import torch


def different_class_permutation(labels: torch.Tensor, attempts: int = 32):
    """Return the candidate permutation with the most different-label pairs."""
    best_perm, best_valid = None, None
    best_count = -1
    for _ in range(attempts):
        perm = torch.randperm(labels.numel(), device=labels.device)
        valid = labels != labels[perm]
        count = int(valid.sum())
        if count > best_count:
            best_perm, best_valid, best_count = perm, valid, count
        if count == labels.numel():
            break
    return best_perm, best_valid


def mix_after_layer2(model, images: torch.Tensor, labels: torch.Tensor, alpha: float = 2.0):
    all_hidden = model.forward_to_layer2(images)
    perm, valid = different_class_permutation(labels)
    if not valid.any():
        raise RuntimeError("A batch with fewer than two represented classes cannot form data placeholders")
    hidden = all_hidden[valid]
    paired = all_hidden[perm[valid]]
    concentration = torch.tensor(alpha, device=images.device, dtype=torch.float32)
    lam = torch.distributions.Beta(concentration, concentration).sample()
    mixed = lam.to(hidden.dtype) * hidden + (1.0 - lam).to(hidden.dtype) * paired
    return mixed, lam, valid
