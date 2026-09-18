"""
Dataset loading, splitting, subset selection, and feature caching/training
utilities for Task 1.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
import torchvision.transforms.functional as TF
from torchvision.datasets import STL10

from .config import (
    CACHE_DIR, DATA_DIR, DEVICE, IMG_SIZE, NUM_WORKERS, SEED, STL10_CLASSES,
    USE_CUDA, autocast_ctx, make_rng, save_json,
)


# ---------------------------------------------------------------------------
# Dataset wrapper: returns (image[3,224,224] in [0,1] float, label, image_id)
# ---------------------------------------------------------------------------
class STL10Wrapped(Dataset):
    """
    Wraps torchvision's STL10, resizing to 224x224 and returning plain
    [0,1] float RGB tensors (NOT normalized — normalization is backbone-
    specific and applied later). image_id is a stable string
    "<split>-<index>" used as a permutation/caching key.
    """

    def __init__(self, split: str, download: bool = True):
        assert split in ("train", "test")
        self.split = split
        self.base = STL10(root=str(DATA_DIR), split=split, download=download)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int):
        img, label = self.base[index]  # PIL image, int label
        img = TF.resize(img, [IMG_SIZE, IMG_SIZE], antialias=True)
        img_t = TF.to_tensor(img)  # [3, H, W] in [0,1]
        image_id = f"{self.split}-{index}"
        return img_t, int(label), image_id


def get_datasets(download: bool = True) -> Tuple[STL10Wrapped, STL10Wrapped]:
    train_ds = STL10Wrapped("train", download=download)
    test_ds = STL10Wrapped("test", download=download)
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Stratified 80/20 split of the official STL-10 train partition
# ---------------------------------------------------------------------------
def stratified_split(dataset: STL10Wrapped, train_frac: float = 0.8, seed_name: str = "stratified_split"):
    labels = np.array(dataset.base.labels)
    rng = make_rng(seed_name)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        n_train = int(round(len(c_idx) * train_frac))
        train_idx.extend(c_idx[:n_train].tolist())
        val_idx.extend(c_idx[n_train:].tolist())
    rng.shuffle(np.array(train_idx))  # order doesn't matter for a Subset, kept for clarity
    return Subset(dataset, sorted(train_idx)), Subset(dataset, sorted(val_idx))


# ---------------------------------------------------------------------------
# Class-balanced 500-image eval subset from the official test partition
# ---------------------------------------------------------------------------
def select_eval_subset(
    test_ds: STL10Wrapped, total: int = 500, seed_name: str = "eval_subset"
) -> Dict:
    labels = np.array(test_ds.base.labels)
    num_classes = len(STL10_CLASSES)
    per_class_target = total // num_classes
    remainder = total - per_class_target * num_classes  # distributed to first classes if not divisible

    rng = make_rng(seed_name)
    selected = []
    shortfall_log = {}
    for c in range(num_classes):
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        target = per_class_target + (1 if c < remainder else 0)
        take = min(target, len(c_idx))
        if take < target:
            shortfall_log[STL10_CLASSES[c]] = {"requested": target, "available": len(c_idx), "used": take}
        selected.extend(c_idx[:take].tolist())

    selected = sorted(selected)
    image_ids = [f"test-{i}" for i in selected]
    result = {
        "seed": SEED,
        "total_requested": total,
        "total_selected": len(selected),
        "per_class_target": per_class_target,
        "indices": selected,
        "image_ids": image_ids,
        "labels": [int(labels[i]) for i in selected],
        "shortfall_log": shortfall_log,
    }
    return result


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------
def _cache_key(backbone_name: str, condition: str, image_ids: List[str]) -> str:
    ids_hash = hashlib.sha1("|".join(image_ids).encode()).hexdigest()[:16]
    return f"feat__{backbone_name}__{condition}__{ids_hash}.pt"


def _image_cache_key(backbone_name: str, condition: str, image_id: str) -> str:
    image_hash = hashlib.sha1(image_id.encode()).hexdigest()[:16]
    return f"feat__{backbone_name}__{condition}__image_{image_hash}.pt"


def _bundle_for_ids(cached: Dict, image_ids: List[str]) -> Optional[Dict]:
    id_to_row = {image_id: row for row, image_id in enumerate(cached["image_ids"])}
    if not all(image_id in id_to_row for image_id in image_ids):
        return None
    rows = [id_to_row[image_id] for image_id in image_ids]
    return {
        "features": cached["features"][rows],
        "labels": [cached["labels"][row] for row in rows],
        "image_ids": image_ids,
    }


def load_cached_features(backbone_name: str, condition: str, image_ids: List[str]) -> Optional[Dict]:
    # Keep reading the original aggregate cache format for existing results.
    exact_path = CACHE_DIR / _cache_key(backbone_name, condition, image_ids)
    if exact_path.exists():
        return torch.load(exact_path, map_location="cpu")

    # New caches are one record per image, allowing arbitrary subsets and order.
    image_bundles = []
    for image_id in dict.fromkeys(image_ids):
        path = CACHE_DIR / _image_cache_key(backbone_name, condition, image_id)
        if not path.exists():
            image_bundles = []
            break
        image_bundles.append(torch.load(path, map_location="cpu"))
    if image_bundles:
        by_id = {bundle["image_ids"][0]: bundle for bundle in image_bundles}
        return {
            "features": torch.cat([by_id[image_id]["features"] for image_id in image_ids]),
            "labels": [by_id[image_id]["labels"][0] for image_id in image_ids],
            "image_ids": image_ids,
        }

    # Also allow a subset lookup from any old aggregate cache for this
    # backbone/condition, which makes the change usable without recomputation.
    prefix = f"feat__{backbone_name}__{condition}__"
    for path in CACHE_DIR.glob(f"{prefix}*.pt"):
        cached = torch.load(path, map_location="cpu")
        bundle = _bundle_for_ids(cached, image_ids)
        if bundle is not None:
            return bundle
    return None


def save_cached_features(backbone_name: str, condition: str, image_ids: List[str], features: torch.Tensor, labels: List[int]):
    features = features.cpu()
    for row, image_id in enumerate(image_ids):
        path = CACHE_DIR / _image_cache_key(backbone_name, condition, image_id)
        torch.save(
            {
                "features": features[row:row + 1],
                "labels": [labels[row]],
                "image_ids": [image_id],
            },
            path,
        )
    return CACHE_DIR


@torch.no_grad()
def extract_features(
    backbone,
    images_iterable_loader: DataLoader,
    condition: str,
    cache: bool = True,
) -> Dict:
    """
    Runs `backbone.forward_features` over a DataLoader yielding
    (images[N,3,224,224] in [0,1], labels[N], image_ids[N-tuple]).
    Applies the given intervention function BEFORE calling forward_features
    if one is baked into the loader/dataset already (this function assumes
    images arriving here are already the desired condition's pixels).
    Caches each feature to disk keyed by (backbone, condition, image-id), so
    later requests can retrieve arbitrary subsets in any order.
    """
    all_ids: List[str] = []
    for _, _, ids in images_iterable_loader:
        all_ids.extend(list(ids))

    cached = load_cached_features(backbone.name, condition, all_ids) if cache else None
    if cached is not None:
        return cached

    feats, labels, ids_out = [], [], []
    for imgs, lbls, ids in images_iterable_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        f = backbone.forward_features(imgs)
        feats.append(f.cpu())
        labels.extend([int(x) for x in lbls])
        ids_out.extend(list(ids))
    feats = torch.cat(feats, dim=0)

    if cache:
        save_cached_features(backbone.name, condition, ids_out, feats, labels)
    return {"features": feats, "labels": labels, "image_ids": ids_out}


def make_loader(dataset, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=USE_CUDA,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Linear classifier head training on cached (frozen) features
# ---------------------------------------------------------------------------
class LinearHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_head(
    train_feats: torch.Tensor,
    train_labels: List[int],
    val_feats: torch.Tensor,
    val_labels: List[int],
    in_features: int,
    num_classes: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 5,
    seed: int = SEED,
    batch_size: int = 256,
) -> Tuple[LinearHead, Dict]:
    torch.manual_seed(seed)
    head = LinearHead(in_features, num_classes).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_feats = train_feats.to(DEVICE)
    val_feats = val_feats.to(DEVICE)
    train_labels_t = torch.tensor(train_labels, device=DEVICE, dtype=torch.long)
    val_labels_t = torch.tensor(val_labels, device=DEVICE, dtype=torch.long)

    use_scaler = USE_CUDA and not torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    n = train_feats.shape[0]
    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    g = torch.Generator(device="cpu").manual_seed(seed)

    epoch_bar = tqdm(range(max_epochs), desc="Training linear head", unit="epoch")
    for epoch in epoch_bar:
        head.train()
        perm = torch.randperm(n, generator=g)
        epoch_loss = 0.0
        batch_bar = tqdm(
            range(0, n, batch_size),
            desc=f"Epoch {epoch + 1}/{max_epochs}",
            unit="batch",
            leave=False,
        )
        for start in batch_bar:
            idx = perm[start:start + batch_size]
            xb = train_feats[idx]
            yb = train_labels_t[idx]
            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = head(xb)
                loss = criterion(logits, yb)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            epoch_loss += loss.item() * xb.shape[0]
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        epoch_loss /= n

        head.eval()
        with torch.no_grad(), autocast_ctx():
            val_logits = head(val_feats)
            val_loss = criterion(val_logits, val_labels_t).item()
            val_pred = val_logits.argmax(dim=-1)
            val_acc = (val_pred == val_labels_t).float().mean().item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        epoch_bar.set_postfix(
            train_loss=f"{epoch_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    history["best_val_acc"] = best_val_acc
    history["epochs_run"] = len(history["train_loss"])
    return head, history
