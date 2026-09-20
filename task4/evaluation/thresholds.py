import torch


def calibrate_threshold(validation_unknownness: torch.Tensor, quantile: float = 0.95) -> float:
    """Known-only calibration: accept when unknownness <= returned threshold."""
    return float(torch.quantile(validation_unknownness.float().cpu(), quantile))
