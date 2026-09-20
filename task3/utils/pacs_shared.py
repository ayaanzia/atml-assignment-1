"""PACS helpers factored from Task 2 for exact reuse in Task 3.

The source protocol deliberately resolves and opens only the three source
domains.  The held-out Sketch directory is resolved only by the notebook's
explicit final-evaluation section.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

SOURCE_DOMAINS = ("photo", "art_painting", "cartoon")
NUM_CLASSES = 7
EXPECTED_SOURCE_SIZES = {
    "photo": 1670,
    "art_painting": 2048,
    "cartoon": 2344,
}
DOMAIN_ALIASES = {
    "photo": {"photo"},
    "art_painting": {"artpainting", "art_painting", "art painting"},
    "cartoon": {"cartoon"},
    "sketch": {"sketch"},
}


def _canonical_name(name: str) -> str:
    normalized = name.lower().replace("-", "_").strip()
    compact = normalized.replace("_", "").replace(" ", "")
    for canonical, aliases in DOMAIN_ALIASES.items():
        alias_compact = {value.replace("_", "").replace(" ", "") for value in aliases}
        if normalized in aliases or compact in alias_compact:
            return canonical
    return normalized


def _candidate_roots(root: Path):
    return (
        root,
        root / "pacs_data" / "pacs_data",
        root / "PACS",
        root / "pacs",
        root / "kfold",
        root / "PACS" / "kfold",
        root / "pacs" / "kfold",
    )


def resolve_source_domain_dirs(root: Path) -> Dict[str, Path]:
    """Resolve only source domains, without requiring or opening Sketch."""
    for candidate in _candidate_roots(root):
        if not candidate.is_dir():
            continue
        found = {_canonical_name(path.name): path for path in candidate.iterdir() if path.is_dir()}
        if all(domain in found for domain in SOURCE_DOMAINS):
            return {domain: found[domain] for domain in SOURCE_DOMAINS}
    raise FileNotFoundError(
        "Could not find Photo, Art Painting, and Cartoon. Set PACS_ROOT to "
        "the extracted PACS directory or its parent."
    )


def resolve_domain_dir(root: Path, domain: str) -> Path:
    """Resolve one named domain. Task 3 calls this for Sketch only at final evaluation."""
    for candidate in _candidate_roots(root):
        if not candidate.is_dir():
            continue
        found = {_canonical_name(path.name): path for path in candidate.iterdir() if path.is_dir()}
        if domain in found:
            return found[domain]
    raise FileNotFoundError(f"Could not find PACS domain {domain!r} below {root}.")


def make_transforms():
    """Return the exact Task 2 ImageNet preprocessing pipelines."""
    weights = ResNet18_Weights.IMAGENET1K_V1
    weight_transform = weights.transforms()
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(weight_transform.mean, weight_transform.std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(weight_transform.mean, weight_transform.std),
        ]
    )
    return train_transform, eval_transform


def load_source_protocol(pacs_root: Path, split_path: Path, seed: int):
    """Load and strictly verify Task 2's saved source splits."""
    domain_dirs = resolve_source_domain_dirs(pacs_root)
    train_transform, eval_transform = make_transforms()
    with split_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("seed") != seed:
        raise ValueError(f"Split seed is {manifest.get('seed')}, expected {seed}.")
    if tuple(manifest.get("source_domains", ())) != SOURCE_DOMAINS:
        raise ValueError("Task 2 split manifest has different source domains.")

    source_train_sets, source_val_sets = {}, {}
    class_to_idx = None
    for domain in SOURCE_DOMAINS:
        train_base = datasets.ImageFolder(domain_dirs[domain], transform=train_transform)
        eval_base = datasets.ImageFolder(domain_dirs[domain], transform=eval_transform)
        if len(eval_base) != EXPECTED_SOURCE_SIZES[domain]:
            raise ValueError(
                f"Unexpected {domain} size {len(eval_base)}; expected "
                f"{EXPECTED_SOURCE_SIZES[domain]} original PACS images."
            )
        if class_to_idx is None:
            class_to_idx = eval_base.class_to_idx
        if eval_base.class_to_idx != class_to_idx:
            raise ValueError(f"Class mapping mismatch in {domain}.")
        if [path for path, _ in train_base.samples] != [path for path, _ in eval_base.samples]:
            raise RuntimeError(f"Sample ordering differs across transforms for {domain}.")

        info = manifest["domains"][domain]
        train_idx = list(map(int, info["train_indices"]))
        val_idx = list(map(int, info["val_indices"]))
        if set(train_idx) & set(val_idx):
            raise ValueError(f"{domain} train/validation indices overlap.")
        if sorted(train_idx + val_idx) != list(range(len(eval_base))):
            raise ValueError(f"{domain} split is not a complete partition.")
        base = domain_dirs[domain]
        train_paths = [str(Path(eval_base.samples[i][0]).relative_to(base)) for i in train_idx]
        val_paths = [str(Path(eval_base.samples[i][0]).relative_to(base)) for i in val_idx]
        if train_paths != info["train_paths"] or val_paths != info["val_paths"]:
            raise ValueError(f"{domain} files differ from Task 2's split manifest.")
        source_train_sets[domain] = Subset(train_base, train_idx)
        source_val_sets[domain] = Subset(eval_base, val_idx)

    if class_to_idx != manifest.get("class_to_idx") or len(class_to_idx) != NUM_CLASSES:
        raise ValueError("Class mapping differs from Task 2.")
    return {
        "domain_dirs": domain_dirs,
        "train_sets": source_train_sets,
        "val_sets": source_val_sets,
        "class_to_idx": class_to_idx,
        "manifest": manifest,
        "eval_transform": eval_transform,
    }


class ResNet18Adapter(nn.Module):
    """The exact Task 2 pretrained ResNet-18 plus fresh seven-class head."""

    def __init__(self):
        super().__init__()
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_dim = network.fc.in_features
        network.fc = nn.Identity()
        self.backbone = network
        self.classifier = nn.Linear(self.feature_dim, NUM_CLASSES)

    def forward(self, inputs):
        features = self.backbone(inputs)
        return self.classifier(features), features


def freeze_batchnorm_running_stats(model: nn.Module) -> None:
    """Freeze running statistics while keeping BatchNorm affine terms trainable."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def multi_kernel_mmd(first_features, second_features):
    """Task 2's biased three-RBF MMD estimator with batch-median bandwidth."""
    combined = torch.cat([first_features, second_features], dim=0).float()
    pairwise_sq = torch.cdist(combined, combined, p=2).pow(2)
    nonzero_distances = torch.pdist(combined, p=2).pow(2)
    median_sq = nonzero_distances.median().detach().clamp_min(1e-6)
    kernel = torch.zeros_like(pairwise_sq)
    for multiplier in (0.5, 1.0, 2.0):
        bandwidth = multiplier * median_sq
        kernel = kernel + torch.exp(-pairwise_sq / (2.0 * bandwidth))
    first_count = first_features.shape[0]
    k_aa = kernel[:first_count, :first_count]
    k_bb = kernel[first_count:, first_count:]
    k_ab = kernel[:first_count, first_count:]
    return k_aa.mean() + k_bb.mean() - 2.0 * k_ab.mean()
