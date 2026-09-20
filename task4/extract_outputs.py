"""Batched, AMP output extraction and disk caching."""
from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm


@torch.inference_mode()
def extract_outputs(model, loader, device):
    model.eval()
    logits, features, labels, indices = [], [], [], []
    for images, targets, source_indices in tqdm(loader, desc="Extract", leave=False):
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            batch_logits, batch_features = model(images, return_features=True)
        logits.append(batch_logits.float().cpu())
        features.append(batch_features.float().cpu())
        labels.append(torch.as_tensor(targets).long().cpu())
        indices.append(torch.as_tensor(source_indices).long().cpu())
    return {"logits": torch.cat(logits), "features": torch.cat(features),
            "labels": torch.cat(labels), "indices": torch.cat(indices)}


def extract_or_load(model, loader, device, cache_path, force: bool = False):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    result = extract_outputs(model, loader, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache_path)
    return result
