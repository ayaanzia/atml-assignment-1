"""Shared-diagonal Mahalanobis scoring over cached, unaugmented features."""
import torch


def fit_shared_diagonal_gaussian(features: torch.Tensor, labels: torch.Tensor,
                                 num_classes: int = 10, eps: float = 1e-6):
    features = features.float()
    labels = labels.long()
    means = torch.stack([features[labels == c].mean(0) for c in range(num_classes)])
    residual = features - means[labels]
    # Maximum-likelihood shared within-class diagonal covariance.
    variance = residual.square().mean(0).clamp_min(eps)
    return means, variance


def mahalanobis_unknownness(features: torch.Tensor, means: torch.Tensor,
                            variance: torch.Tensor, chunk_size: int = 4096):
    outputs = []
    for chunk in features.float().split(chunk_size):
        distances = ((chunk[:, None, :] - means[None, :, :]).square() /
                     variance[None, None, :]).sum(2)
        outputs.append(distances.amin(1))
    return torch.cat(outputs)
