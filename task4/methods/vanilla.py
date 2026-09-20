"""Vanilla closed-set training with validation checkpoint selection."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from task4.models import resnet18_cifar


def seed_everything(seed: int = 6304) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.inference_mode()
def known_accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
        if logits.shape[1] > 10:
            logits = logits[:, :10]
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.numel()
    return correct / total


def train_closed_set(model, train_loader, val_loader, config: dict, checkpoint_path, device):
    seed_everything(int(config.get("seed", 6304)))
    model = model.to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(config["learning_rate"]),
        momentum=float(config["momentum"]), weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(config["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    history = []
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(config["epochs"])):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}", leave=True)
        for step, (images, targets, _) in enumerate(pbar, 1):
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / step:.4f}")
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


def train_vanilla(train_loader, val_loader, config: dict, checkpoint_path, device):
    seed_everything(int(config.get("seed", 6304)))
    return train_closed_set(resnet18_cifar(10), train_loader, val_loader, config, checkpoint_path, device)
