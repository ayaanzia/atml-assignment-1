"""PROSER fine-tuning following Zhou, Ye & Zhan (CVPR 2021)."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from task4.models import resnet18_cifar
from .manifold_mixup import mix_after_layer2
from .vanilla import known_accuracy, seed_everything


class PROSER(nn.Module):
    def __init__(self, num_known: int = 10, num_dummy: int = 5):
        super().__init__()
        self.num_known = num_known
        self.num_dummy = num_dummy
        self.network = resnet18_cifar(num_known + num_dummy)

    @classmethod
    def from_vanilla_checkpoint(cls, checkpoint_path, num_dummy: int = 5,
                                map_location="cpu", seed: int = 6304):
        seed_everything(seed)
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        source = checkpoint["model"]
        model = cls(10, num_dummy)
        target = model.network.state_dict()
        for key, value in source.items():
            if key not in ("fc.weight", "fc.bias"):
                target[key].copy_(value)
        target["fc.weight"][:10].copy_(source["fc.weight"])
        target["fc.bias"][:10].copy_(source["fc.bias"])
        model.network.load_state_dict(target)
        return model

    def forward_to_layer2(self, x):
        return self.network.forward_to_layer2(x)

    def features_from_layer2(self, x):
        return self.network.features_from_layer2(x)

    def logits_from_layer2(self, x):
        return self.network.logits_from_layer2(x)

    def forward(self, x, return_features: bool = False):
        return self.network(x, return_features=return_features)


def aggregate_dummy_logits(logits: torch.Tensor, num_known: int = 10) -> torch.Tensor:
    """Paper Eq. 5: retain known logits and only the strongest dummy logit."""
    return torch.cat((logits[:, :num_known], logits[:, num_known:].amax(1, keepdim=True)), dim=1)


def classifier_placeholder_loss(logits: torch.Tensor, targets: torch.Tensor,
                                num_known: int = 10, beta: float = 1.0):
    aggregated = aggregate_dummy_logits(logits, num_known)
    closed_loss = F.cross_entropy(aggregated, targets)
    masked = aggregated.clone()
    masked.scatter_(1, targets[:, None], torch.finfo(masked.dtype).min)
    dummy_targets = torch.full_like(targets, num_known)
    boundary_loss = F.cross_entropy(masked, dummy_targets)
    return closed_loss + beta * boundary_loss, closed_loss, boundary_loss


def data_placeholder_loss(logits: torch.Tensor, num_known: int = 10):
    aggregated = aggregate_dummy_logits(logits, num_known)
    return F.cross_entropy(aggregated, torch.full(
        (len(aggregated),), num_known, dtype=torch.long, device=aggregated.device))


def train_proser(model, train_loader, val_loader, config: dict, checkpoint_path, device):
    seed_everything(int(config.get("seed", 6304)))
    model = model.to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config["learning_rate"]),
                                momentum=float(config["momentum"]),
                                weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(config["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy, history = -1.0, []

    for epoch in range(int(config["epochs"])):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}", leave=True)
        for step, (images, targets, _) in enumerate(pbar, 1):
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            half = len(images) // 2
            if half == 0:
                continue
            class_images, class_targets = images[:half], targets[:half]
            mix_images, mix_targets = images[half:2 * half], targets[half:2 * half]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                class_logits = model(class_images)
                class_loss, closed_loss, boundary_loss = classifier_placeholder_loss(
                    class_logits, class_targets, model.num_known, float(config["beta"]))
                mixed_hidden, _, _ = mix_after_layer2(
                    model, mix_images, mix_targets, float(config["mixup_alpha"]))
                mix_logits = model.logits_from_layer2(mixed_hidden)
                mix_loss = data_placeholder_loss(mix_logits, model.num_known)
                loss = class_loss + float(config["gamma"]) * mix_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / step:.4f}", closed=f"{closed_loss.item():.3f}",
                             boundary=f"{boundary_loss.item():.3f}", mix=f"{mix_loss.item():.3f}")
        val_accuracy = known_accuracy(model, val_loader, device)
        history.append({"epoch": epoch + 1, "loss": running_loss / step, "val_accuracy": val_accuracy})
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "val_accuracy": val_accuracy, "config": dict(config)}, checkpoint_path)
        scheduler.step()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model, history


def placeholder_unknownness(logits: torch.Tensor, num_known: int = 10,
                            temperature: float = 1024.0) -> torch.Tensor:
    """Reference ΔP score: p(strongest dummy) - max p(known)."""
    probabilities = aggregate_dummy_logits(logits, num_known).float().div(temperature).softmax(1)
    return probabilities[:, -1] - probabilities[:, :num_known].amax(1)
